import logging
from collections.abc import Callable
from typing import Any, Optional, Union

import keras
from keras.layers import Layer
from keras.models import Model

from decomon.core import (
    BoxDomain,
    ConvertMethod,
    PerturbationDomain,
    Propagation,
    Slope,
)
from decomon.layers import DecomonLayer
from decomon.layers.convert import to_decomon
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
    preprocess_backward_bounds,
    preprocess_layer,
    split_activation,
)

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
    backward_bounds: Optional[list[list[keras.KerasTensor]]] = None,
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
        backward_bounds: backward bounds to propagate, see `preprocess_backward_bounds()` for conventions
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
            slope=slope,
            forward_output_map=forward_output_map,
            forward_layer_map=forward_layer_map,
            **kwargs,
        )

    # Update output for final_ibp and final_affine
    ...

    # build decomon model
    return output


def clone(
    model: Model,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    perturbation_domain: Optional[PerturbationDomain] = None,
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    backward_bounds: Optional[Union[keras.KerasTensor, list[keras.KerasTensor], list[list[keras.KerasTensor]]]] = None,
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
        backward_bounds: backward bounds to propagate, see `preprocess_backward_bounds()` for conventions
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

    backward_bounds = preprocess_backward_bounds(backward_bounds=backward_bounds, nb_model_outputs=len(model.outputs))

    perturbation_domain_input = generate_perturbation_domain_input(
        model=model, perturbation_domain=perturbation_domain, name=f"perturbation_domain_input_{model_name}"
    )

    output = convert(
        model=model,
        perturbation_domain_input=perturbation_domain_input,
        perturbation_domain=perturbation_domain,
        method=method,
        backward_bounds=backward_bounds,
        layer_fn=layer_fn,
        slope=slope,
        forward_output_map=forward_output_map,
        forward_layer_map=forward_layer_map,
        final_ibp=final_ibp,
        final_affine=final_affine,
        **kwargs,
    )

    # full linear model? => batch independent output
    if any([not isinstance(o, keras.KerasTensor) for o in output]):
        logger.warning(
            "Some propagated bounds have been eagerly computed, being independent from batch. "
            "This should only be possible if the keras model is fully affine. -"
            "We will make them artificially depend on perturbation input in order to create the DecomonModel."
        )
        # Insert batch axis and repeat it to get the correct batchsize
        output = LinkToPerturbationDomainInput()([perturbation_domain_input] + output)

    return DecomonModel(
        inputs=[perturbation_domain_input],
        outputs=output,
        perturbation_domain=perturbation_domain,
        method=method,
        ibp=final_ibp,
        affine=final_affine,
    )
