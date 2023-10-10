"""Module for DecomonSequential.

It inherits from keras Sequential class.

"""
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from keras_core.layers import Layer
from keras_core.models import Model
from keras_core.src.utils.python_utils import to_list

from decomon.core import BoxDomain, PerturbationDomain, Slope
from decomon.layers.convert import to_decomon
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.utils import (
    get_depth_dict,
    get_inner_layers,
    get_input_dim,
    prepare_inputs_for_layer,
    wrap_outputs_from_layer_in_list,
)

OutputMapKey = Union[str, int]
OutputMapVal = Union[List[tf.Tensor], "OutputMapDict"]
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
    input_tensors: List[tf.Tensor],
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
    joint: bool = True,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], LayerMapDict, OutputMapDict]:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

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

    f_output = convert_forward_functional_model(
        model=model,
        input_tensors=input_tensors,
        layer_fn=layer_fn_to_list,
        softmax_to_linear=softmax_to_linear,
        joint=joint,
    )

    return f_output


def convert_forward_functional_model(
    model: Model,
    layer_fn: Callable[[Layer], List[Layer]],
    input_tensors: List[tf.Tensor],
    softmax_to_linear: bool = True,
    count: int = 0,
    joint: bool = True,
    output_map: Optional[OutputMapDict] = None,
    layer_map: Optional[LayerMapDict] = None,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], LayerMapDict, OutputMapDict]:
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)

    # create input tensors

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    if output_map is None:
        output_map = {}
    if layer_map is None:
        layer_map = {}
    output: List[tf.Tensor] = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            layer = node.outbound_layer
            parents = node.parent_nodes
            if id(node) in output_map.keys() and joint:
                continue
            if len(parents):
                output = []
                for parent in parents:
                    output += output_map[id(parent)]

            if isinstance(layer, Model):
                _, output, layer_map_submodel, output_map_submodel = convert_forward_functional_model(
                    model=layer,
                    input_tensors=output,
                    layer_fn=layer_fn,
                    softmax_to_linear=softmax_to_linear,
                    count=count,
                )
                count = count + get_inner_layers(layer)
                layer_map[id(node)] = layer_map_submodel
                output_map[id(node)] = output_map_submodel
            else:
                converted_layers = layer_fn(layer)
                for converted_layer in converted_layers:
                    converted_layer._name = f"{converted_layer.name}_{count}"
                    count += 1
                    output = converted_layer(prepare_inputs_for_layer(output))
                    if len(converted_layers) > 1:
                        output_map[f"{id(node)}_{converted_layer.name}"] = wrap_outputs_from_layer_in_list(output)
                layer_map[id(node)] = converted_layers
            output_map[id(node)] = wrap_outputs_from_layer_in_list(output)

    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.layer.name for tensor in to_list(model.output)]
    for output_name in output_names:
        for node in output_nodes:
            if node.outbound_layer.name == output_name:
                output += output_map[id(node)]

    return input_tensors, output, layer_map, output_map
