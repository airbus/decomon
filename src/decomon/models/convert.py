from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
from keras.layers import InputLayer, Layer
from keras.models import Model

from decomon.backward_layers.core import BackwardLayer
from decomon.core import BoxDomain, PerturbationDomain, Slope, get_mode
from decomon.layers.convert import to_decomon
from decomon.models.backward_cloning import convert_backward
from decomon.models.forward_cloning import (
    LayerMapDict,
    OutputMapDict,
    convert_forward,
    convert_forward_functional_model,
)
from decomon.models.models import DecomonModel
from decomon.models.utils import (
    Convert2Mode,
    ConvertMethod,
    FeedDirection,
    check_model2convert_inputs,
    convert_deellip_to_keras,
    ensure_functional_model,
    get_direction,
    get_ibp_affine_from_method,
    get_input_tensors,
    is_input_node,
    preprocess_layer,
    split_activation,
)


def _clone_keras_model(model: Model, layer_fn: Callable[[Layer], List[Layer]]) -> Model:
    if model.inputs is None:
        raise ValueError("model.inputs must be not None. You should call the model on a batch of data.")

    # ensure the model is functional or convert to it if a sequential one
    model = ensure_functional_model(model)

    # initialize output_map and layer_map to avoid
    #   - recreating input layers
    #   - and converting input layers and have a cycle around them
    output_map: OutputMapDict = {}
    layer_map: LayerMapDict = {}
    for depth, nodes in model._nodes_by_depth.items():
        for node in nodes:
            if is_input_node(node):
                output_map[id(node)] = node.output_tensors
                layer_map[id(node)] = node.operation

    _, output, _, _ = convert_forward_functional_model(
        model=model,
        input_tensors=model.inputs,
        softmax_to_linear=False,
        layer_fn=layer_fn,
        output_map=output_map,
        layer_map=layer_map,
    )

    return Model(
        inputs=model.inputs,
        outputs=output,
    )


def split_activations_in_keras_model(
    model: Model,
) -> Model:
    return _clone_keras_model(model=model, layer_fn=split_activation)


def convert_deellip_layers_in_keras_model(
    model: Model,
) -> Model:
    return _clone_keras_model(model=model, layer_fn=_convert_deellip_to_keras)


def _convert_deellip_to_keras(layer: Layer) -> List[Layer]:
    return [convert_deellip_to_keras(layer=layer)]


def preprocess_keras_model(
    model: Model,
) -> Model:
    return _clone_keras_model(model=model, layer_fn=preprocess_layer)


# create status
def convert(
    model: Model,
    input_tensors: List[keras.KerasTensor],
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    ibp: bool = False,
    affine: bool = False,
    back_bounds: Optional[List[keras.KerasTensor]] = None,
    layer_fn: Callable[..., Layer] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    input_dim: int = -1,
    perturbation_domain: Optional[PerturbationDomain] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    shared: bool = True,
    softmax_to_linear: bool = True,
    finetune_forward: bool = False,
    finetune_backward: bool = False,
    final_ibp: bool = False,
    final_affine: bool = False,
    **kwargs: Any,
) -> Tuple[
    List[keras.KerasTensor],
    List[keras.KerasTensor],
    Union[LayerMapDict, Dict[int, BackwardLayer]],
    Optional[OutputMapDict],
]:
    if back_bounds is None:
        back_bounds = []
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if finetune:
        finetune_forward = True
        finetune_backward = True

    if isinstance(method, str):
        method = ConvertMethod(method.lower())

    # prepare the Keras Model: split non-linear activation functions into separate Activation layers
    model = preprocess_keras_model(model)

    layer_map: Union[LayerMapDict, Dict[int, BackwardLayer]]

    if method != ConvertMethod.CROWN:
        input_tensors, output, layer_map, forward_map = convert_forward(
            model=model,
            input_tensors=input_tensors,
            layer_fn=layer_fn,
            slope=slope,
            input_dim=input_dim,
            dc_decomp=False,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            finetune=finetune_forward,
            shared=shared,
            softmax_to_linear=softmax_to_linear,
            back_bounds=back_bounds,
        )

    if get_direction(method) == FeedDirection.BACKWARD:
        input_tensors, output, layer_map, forward_map = convert_backward(
            model=model,
            input_tensors=input_tensors,
            back_bounds=back_bounds,
            slope=slope,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            finetune=finetune_backward,
            forward_map=forward_map,
            final_ibp=final_ibp,
            final_affine=final_affine,
            **kwargs,
        )
    else:
        # check final_ibp and final_affine
        mode_from = get_mode(ibp, affine)
        mode_to = get_mode(final_ibp, final_affine)
        output = Convert2Mode(mode_from=mode_from, mode_to=mode_to, perturbation_domain=perturbation_domain)(output)

    # build decomon model
    return input_tensors, output, layer_map, forward_map


def clone(
    model: Model,
    layer_fn: Callable[..., Layer] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    perturbation_domain: Optional[PerturbationDomain] = None,
    method: Union[str, ConvertMethod] = ConvertMethod.CROWN,
    back_bounds: Optional[List[keras.KerasTensor]] = None,
    finetune: bool = False,
    shared: bool = True,
    finetune_forward: bool = False,
    finetune_backward: bool = False,
    extra_inputs: Optional[List[keras.KerasTensor]] = None,
    to_keras: bool = True,
    final_ibp: Optional[bool] = None,
    final_affine: Optional[bool] = None,
    **kwargs: Any,
) -> DecomonModel:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    if back_bounds is None:
        back_bounds = []
    if extra_inputs is None:
        extra_inputs = []
    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    ibp, affine = get_ibp_affine_from_method(method)
    if final_ibp is None:
        final_ibp = ibp
    if final_affine is None:
        final_affine = affine

    if isinstance(method, str):
        method = ConvertMethod(method.lower())

    if not to_keras:
        raise NotImplementedError("Only convert to Keras for now.")

    if finetune:
        finetune_forward = True
        finetune_backward = True

    # Check hypotheses: functional model + 1 flattened input
    model = ensure_functional_model(model)
    check_model2convert_inputs(model)

    z_tensor, input_tensors = get_input_tensors(
        model=model,
        perturbation_domain=perturbation_domain,
        ibp=ibp,
        affine=affine,
    )

    _, output, _, _ = convert(
        model,
        layer_fn=layer_fn,
        slope=slope,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
        method=method,
        ibp=ibp,
        affine=affine,
        input_dim=-1,
        perturbation_domain=perturbation_domain,
        finetune=finetune,
        shared=shared,
        softmax_to_linear=True,
        layer_map={},
        forward_map={},
        finetune_forward=finetune_forward,
        finetune_backward=finetune_backward,
        final_ibp=final_ibp,
        final_affine=final_affine,
    )

    back_bounds_from_inputs = [elem for elem in back_bounds if isinstance(elem._keras_history.operation, InputLayer)]

    return DecomonModel(
        inputs=[z_tensor] + back_bounds_from_inputs + extra_inputs,
        outputs=output,
        perturbation_domain=perturbation_domain,
        dc_decomp=False,
        method=method,
        ibp=final_ibp,
        affine=final_affine,
        finetune=finetune,
        shared=shared,
        backward_bounds=(len(back_bounds) > 0),
    )
