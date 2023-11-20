"""Module for DecomonSequential.

It inherits from keras Sequential class.

"""
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
from keras.layers import InputLayer, Layer
from keras.models import Model
from keras.src.utils.python_utils import to_list

from decomon.core import BoxDomain, PerturbationDomain, Slope
from decomon.layers.convert import to_decomon
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.utils import (
    ensure_functional_model,
    get_depth_dict,
    get_inner_layers,
    get_input_dim,
    prepare_inputs_for_layer,
    wrap_outputs_from_layer_in_list,
)

OutputMapKey = Union[str, int]
OutputMapVal = Union[List[keras.KerasTensor], "OutputMapDict"]
OutputMapDict = Dict[OutputMapKey, OutputMapVal]

LayerMapVal = Union[List[DecomonLayer], "LayerMapDict"]
LayerMapDict = Dict[int, LayerMapVal]


def include_dim_layer_fn(
    layer_fn: Callable[..., Layer],
    input_dim: int,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    ibp: bool = True,
    affine: bool = True,
    finetune: bool = False,
    shared: bool = True,
) -> Callable[[Layer], List[Layer]]:
    """include external parameters inside the translation of a layer to its decomon counterpart

    Args:
        layer_fn
        input_dim
        dc_decomp
        perturbation_domain
        finetune

    Returns:

    """
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    layer_fn_copy = deepcopy(layer_fn)

    if "input_dim" in inspect.signature(layer_fn).parameters:

        def func(layer: Layer) -> List[Layer]:
            return [
                layer_fn_copy(
                    layer,
                    input_dim=input_dim,
                    slope=slope,
                    perturbation_domain=perturbation_domain,
                    dc_decomp=dc_decomp,
                    ibp=ibp,
                    affine=affine,
                    finetune=finetune,
                    shared=shared,
                )
            ]

    else:

        def func(layer: Layer) -> List[Layer]:
            return [layer_fn_copy(layer)]

    return func


def convert_forward(
    model: Model,
    input_tensors: List[keras.KerasTensor],
    layer_fn: Callable[..., Layer] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    input_dim: int = -1,
    dc_decomp: bool = False,
    perturbation_domain: Optional[PerturbationDomain] = None,
    ibp: bool = True,
    affine: bool = True,
    finetune: bool = False,
    shared: bool = True,
    softmax_to_linear: bool = True,
    **kwargs: Any,
) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor], LayerMapDict, OutputMapDict]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    model = ensure_functional_model(model)

    if input_dim == -1:
        input_dim = get_input_dim(model)

    layer_fn_to_list = include_dim_layer_fn(
        layer_fn,
        input_dim=input_dim,
        slope=slope,
        perturbation_domain=perturbation_domain,
        ibp=ibp,
        affine=affine,
        finetune=finetune,
        shared=shared,
    )  # return a list of Decomon layers

    input_tensors, output, layer_map, output_map, _ = convert_forward_functional_model(
        model=model,
        input_tensors=input_tensors,
        layer_fn=layer_fn_to_list,
        softmax_to_linear=softmax_to_linear,
    )

    return input_tensors, output, layer_map, output_map


def convert_forward_functional_model(
    model: Model,
    layer_fn: Callable[[Layer], List[Layer]],
    input_tensors: List[keras.KerasTensor],
    softmax_to_linear: bool = True,
    count: int = 0,
    output_map: Optional[OutputMapDict] = None,
    layer_map: Optional[LayerMapDict] = None,
    layer2layer_map: Optional[Dict[int, List[Layer]]] = None,
) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor], LayerMapDict, OutputMapDict, Dict[int, List[Layer]]]:
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)

    # ensure the model is functional (to be able to use get_depth_dict), convert sequential ones into functional ones
    model = ensure_functional_model(model)

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    if output_map is None:
        output_map = {}
    if layer_map is None:
        layer_map = {}
    if layer2layer_map is None:
        layer2layer_map = {}
    output: List[keras.KerasTensor] = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            layer = node.operation
            parents = node.parent_nodes
            if id(node) in output_map.keys():
                continue
            if len(parents):
                output = []
                for parent in parents:
                    output += output_map[id(parent)]

            if isinstance(layer, InputLayer):
                # no conversion, no transformation for output (instead of trying to pass in identity layer)
                layer_map[id(node)] = layer
            elif isinstance(layer, Model):
                (
                    _,
                    output,
                    layer_map_submodel,
                    output_map_submodel,
                    layer2layer_map_submodel,
                ) = convert_forward_functional_model(
                    model=layer,
                    input_tensors=output,
                    layer_fn=layer_fn,
                    softmax_to_linear=softmax_to_linear,
                    count=count,
                    layer2layer_map=layer2layer_map,
                )
                count = count + get_inner_layers(layer)
                layer_map.update(layer_map_submodel)
                output_map.update(output_map_submodel)
                layer2layer_map.update(layer2layer_map_submodel)
                layer_map[id(node)] = layer_map_submodel
            else:
                if id(layer) in layer2layer_map:
                    # avoid converting twice layers that are shared by several nodes
                    converted_layers = layer2layer_map[id(layer)]
                else:
                    converted_layers = layer_fn(layer)
                    layer2layer_map[id(layer)] = converted_layers
                for converted_layer in converted_layers:
                    converted_layer.name = f"{converted_layer.name}_{count}"
                    count += 1
                    output = converted_layer(prepare_inputs_for_layer(output))
                    if len(converted_layers) > 1:
                        output_map[f"{id(node)}_{converted_layer.name}"] = wrap_outputs_from_layer_in_list(output)
                layer_map[id(node)] = converted_layers
            output_map[id(node)] = wrap_outputs_from_layer_in_list(output)

    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.operation.name for tensor in to_list(model.output)]
    for output_name in output_names:
        for node in output_nodes:
            if node.operation.name == output_name:
                output += output_map[id(node)]

    return input_tensors, output, layer_map, output_map, layer2layer_map
