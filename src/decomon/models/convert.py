import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import keras
from keras.layers import Layer
from keras.models import Model

from decomon.constants import ConvertMethod, Propagation, Slope
from decomon.layers import DecomonLayer
from decomon.layers.convert import to_decomon
from decomon.layers.fuse import Fuse
from decomon.layers.input import (
    BackwardInput,
    IdentityInput,
    flatten_backward_bounds,
    has_no_backward_bounds,
)
from decomon.layers.output import ConvertOutput
from decomon.layers.utils.symbolify import LinkToPerturbationDomainInput
from decomon.models.backward_cloning import convert_backward
from decomon.models.forward_cloning import (
    convert_forward,
    convert_forward_functional_model,
)
from decomon.models.models import DecomonModel
from decomon.models.utils import (
    ensure_functional_model,
    generate_perturbation_domain_input,
    get_final_ibp_affine_from_method,
    get_ibp_affine_from_method,
    is_input_node,
    method2propagation,
    preprocess_layer,
    split_activation,
)
from decomon.perturbation_domain import BoxDomain, PerturbationDomain

logger = logging.getLogger(__name__)


def _clone_keras_model(model: Model, layer_fn: Callable[[Layer], list[Layer]]) -> Model:
    if model.inputs is None:
        raise ValueError("model.inputs must be not None. You should call the model on a batch of data.")

    # ensure the model is functional or convert to it if a sequential one
    model = ensure_functional_model(model)

    # initialize output_map and layer_map to avoid
    #   - recreating input layers
    #   - and converting input layers and have a cycle around them
    output_map: dict[int, list[keras.KerasTensor]] = {}
    for depth, nodes in model._nodes_by_depth.items():
        for node in nodes:
            if is_input_node(node):
                output_map[id(node)] = node.output_tensors

    output = convert_forward_functional_model(
        model=model,
        input_tensors=model.inputs,
        layer_fn=layer_fn,
        output_map=output_map,
    )

    return Model(
        inputs=model.inputs,
        outputs=output,
    )


def split_activations_in_keras_model(
    model: Model,
) -> Model:
    return _clone_keras_model(model=model, layer_fn=split_activation)


def preprocess_keras_model(
    model: Model,
) -> Model:
    return _clone_keras_model(model=model, layer_fn=preprocess_layer)


# create status
def convert(
    model: Model,
    perturbation_domain_input: keras.KerasTensor,
    perturbation_domain: PerturbationDomain,
    method: ConvertMethod = ConvertMethod.CROWN,
    backward_bounds: Optional[list[keras.KerasTensor]] = None,
    from_linear_backward_bounds: Union[bool, list[bool]] = False,
    layer_fn: Callable[..., DecomonLayer] = to_decomon,
    slope: Slope = Slope.V_SLOPE,
    forward_output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    forward_layer_map: Optional[dict[int, DecomonLayer]] = None,
    final_ibp: bool = False,
    final_affine: bool = True,
    **kwargs: Any,
) -> list[keras.KerasTensor]:
    """

    Args:
        model: keras model to convert
        perturbation_domain_input: perturbation domain input
        perturbation_domain: perturbation domain type on keras model input
        method: method used to convert the model to a decomon model. See `ConvertMethod`.
        backward_bounds: backward bounds to propagate, concatenation of backward bounds for each keras model output
        from_linear_backward_bounds: specify if backward_bounds come from a linear model (=> no batchsize + upper == lower)
            if a boolean, flag for each backward bound, else a list of boolean, one per keras model output.
        layer_fn: callable converting a layer and a model_output_shape into a decomon layer
        slope: slope used by decomon activation layers
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
            To be recomputed if empty and needed by the method.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
            To be recomputed if empty and needed by the method.
        final_ibp: specify if final outputs should include constant bounds.
        final_affine: specify if final outputs should include affine bounds.
        **kwargs: keyword arguments to pass to layer_fn

    Returns:
        propagated bounds (concatenated), output of the future decomon model

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if not final_ibp and not final_affine:
        raise ValueError("One of final_ibp and final_affine must be True.")

    if isinstance(from_linear_backward_bounds, bool):
        from_linear_backward_bounds = [from_linear_backward_bounds] * len(model.outputs)

    # prepare the Keras Model: split non-linear activation functions into separate Activation layers
    model = preprocess_keras_model(model)

    # loop over propagations needed
    propagations = method2propagation(method)
    ibp, affine = get_ibp_affine_from_method(method)
    output: list[keras.KerasTensor] = []

    if Propagation.FORWARD in propagations:
        output, forward_output_map, forward_layer_map = convert_forward(
            model=model,
            perturbation_domain_input=perturbation_domain_input,
            layer_fn=layer_fn,
            slope=slope,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            **kwargs,
        )

    if Propagation.BACKWARD in propagations:
        output = convert_backward(
            model=model,
            perturbation_domain_input=perturbation_domain_input,
            perturbation_domain=perturbation_domain,
            layer_fn=layer_fn,
            backward_bounds=backward_bounds,
            from_linear_backward_bounds=from_linear_backward_bounds,
            slope=slope,
            forward_output_map=forward_output_map,
            forward_layer_map=forward_layer_map,
            **kwargs,
        )
        # output updated mode
        affine = True
        ibp = False

    elif backward_bounds is not None:
        # Fuse backward_bounds with forward bounds if method not using backward propagation
        fuse_layer = Fuse(
            ibp_1=ibp,
            affine_1=affine,
            ibp_2=False,
            affine_2=True,
            m1_input_shape=model.inputs[0].shape[1:],
            m_1_output_shapes=[t.shape[1:] for t in model.outputs],
            from_linear_2=from_linear_backward_bounds,
        )
        output = fuse_layer((output, backward_bounds))
        # output updated mode
        affine = fuse_layer.affine_fused
        ibp = fuse_layer.ibp_fused

    # Update output for final_ibp and final_affine
    if final_ibp != ibp or final_affine != affine:
        convert_layer = ConvertOutput(
            ibp_from=ibp,
            affine_from=affine,
            ibp_to=final_ibp,
            affine_to=final_affine,
            perturbation_domain=perturbation_domain,
            model_output_shapes=[t.shape[1:] for t in model.outputs],
        )
        if convert_layer.needs_perturbation_domain_inputs():
            output.append(perturbation_domain_input)
        output = convert_layer(output)

    # build decomon model
    return output


def clone(
    model: Model,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    perturbation_domain: Optional[PerturbationDomain] = None,
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    backward_bounds: Optional[Union[keras.KerasTensor, list[keras.KerasTensor], list[list[keras.KerasTensor]]]] = None,
    from_linear_backward_bounds: Union[bool, list[bool]] = False,
    final_ibp: Optional[bool] = None,
    final_affine: Optional[bool] = None,
    layer_fn: Callable[..., DecomonLayer] = to_decomon,
    forward_output_map: Optional[dict[int, list[keras.KerasTensor]]] = None,
    forward_layer_map: Optional[dict[int, DecomonLayer]] = None,
    **kwargs: Any,
) -> DecomonModel:
    """

    Args:
        model: keras model to convert
        slope: slope used by decomon activation layers
        perturbation_domain: perturbation domain type on keras model input
        method: method used to convert the model to a decomon model. See `ConvertMethod`.
        backward_bounds: backward bounds to propagate, see `BackwardInput` for conventions
            to be fused with ibp + affine bounds computed on the keras model outputs
        from_linear_backward_bounds: specify if backward_bounds come from a linear model (=> no batchsize + upper == lower)
            if a boolean, flag for each backward bound, else a list of boolean, one per keras model output.
        final_ibp: specify if final outputs should include constant bounds.
            Default to False except for forward-ibp and forward-hybrid.
        final_affine: specify if final outputs should include affine bounds.
            Default to True all methods except forward-ibp.
        layer_fn: callable converting a layer and a model_output_shape into a decomon layer
        forward_output_map: forward outputs per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
            To be recomputed if empty and needed by the method.
        forward_layer_map: forward decomon layer per node from a previously performed forward conversion.
            To be used for forward oracle if not empty.
            To be recomputed if empty and needed by the method.
        **kwargs: keyword arguments to pass to layer_fn

    Returns:
        decomon model mapping perturbation domain input and backward bounds
        to ibp + affine bounds (according to final_ibp and final_affine) on model outputs
        fused with the backward bounds

    The resulting DecomonModel have flatten inputs and outputs:
    - inputs: [perturbation_domain_input] + backward_bounds_flattened
        where backward_bounds_flattened is computed from backward_bounds as follows:
        - None -> []
        - single tensor -> [backward_bounds]
        - list of tensors -> backward_bounds
        - list of list of tensors -> flatten: [t for sublist in backward_bounds for t in sublist]

        - single tensor -> [backward_bounds, 0, backward_bounds, 0] * nb_model_outputs
        - list of 2 tensors (upper = lower bounds, for all model outputs) ->  backward_bounds * 2 * nb_model_outputs
        - list of 4 tensors -> backward_bounds * nb_model_outputs
        - list of 4 * nb_model_outputs tensors -> backward_bounds
        - list of list of tensors -> ensure having nb_model_outputs sublists -> flatten: [t for sublist in backward_bounds for t in sublist]

    - outputs: sum_{i} (affine_bounds_from[i] + constant_bounds_from[i])
        being the affine and constant bounds for each output of the keras model, with
        - i: the indice of the model output considered
        - sum_{i}: the concatenation of subsequent lists over i
        - affine_bounds_from[i]: empty if `final_affine` is False
        - constant_bounds_from[i]: empty if `final_ibp` is False

    """
    # Store model name (before converting to functional)
    model_name = model.name

    # Check hypotheses: functional model + 1 flattened input
    model = ensure_functional_model(model)
    if len(model.inputs) > 1:
        raise ValueError("The model must have only 1 input to be converted.")

    # Args preprocessing
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    default_final_ibp, default_final_affine = get_final_ibp_affine_from_method(method)
    if final_ibp is None:
        final_ibp = default_final_ibp
    if final_affine is None:
        final_affine = default_final_affine

    if isinstance(method, str):
        method = ConvertMethod(method.lower())

    # preprocess backward_bounds
    backward_bounds_flattened: Optional[list[keras.KerasTensor]]
    backward_bounds_for_convert: Optional[list[keras.KerasTensor]]
    if has_no_backward_bounds(backward_bounds):
        backward_bounds_flattened = None
        backward_bounds_for_convert = None
    else:
        if isinstance(from_linear_backward_bounds, bool):
            from_linear_backward_bounds = [from_linear_backward_bounds] * len(model.outputs)
        # flatten backward bounds
        backward_bounds_flattened = flatten_backward_bounds(backward_bounds)
        # prepare for convert: ensure having 4 * nb_model_outputs tensors
        backward_bounds_for_convert = BackwardInput(
            model_output_shapes=[t.shape[1:] for t in model.outputs], from_linear=from_linear_backward_bounds
        )(backward_bounds_flattened)

    perturbation_domain_input = generate_perturbation_domain_input(
        model=model, perturbation_domain=perturbation_domain, name=f"perturbation_domain_input_{model_name}"
    )

    output = convert(
        model=model,
        perturbation_domain_input=perturbation_domain_input,
        perturbation_domain=perturbation_domain,
        method=method,
        backward_bounds=backward_bounds_for_convert,
        from_linear_backward_bounds=from_linear_backward_bounds,
        layer_fn=layer_fn,
        slope=slope,
        forward_output_map=forward_output_map,
        forward_layer_map=forward_layer_map,
        final_ibp=final_ibp,
        final_affine=final_affine,
        **kwargs,
    )

    # model ~ identity: in backward propagation, output can still be empty => diag representation of identity
    if len(output) == 0:  # identity bounds propagated as is (only if successive identity layers)
        output = IdentityInput(perturbation_domain=perturbation_domain)(perturbation_domain_input)

    # full linear model? => batch independent output
    if any([not isinstance(o, keras.KerasTensor) for o in output]):
        logger.warning(
            "Some propagated bounds have been eagerly computed, being independent from batch. "
            "This should only be possible if the keras model is fully affine. -"
            "We will make them artificially depend on perturbation input in order to create the DecomonModel."
        )
        # Insert batch axis and repeat it to get the correct batchsize
        output = LinkToPerturbationDomainInput()([perturbation_domain_input] + output)

    decomon_inputs = [perturbation_domain_input]
    if backward_bounds_flattened is not None:
        decomon_inputs += backward_bounds_flattened

    return DecomonModel(
        inputs=decomon_inputs,
        outputs=output,
        perturbation_domain=perturbation_domain,
        method=method,
        ibp=final_ibp,
        affine=final_affine,
    )
