"""Module for MonotonicSequential.

It inherits from keras Sequential class.

"""
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import to_list

from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_layers import to_decomon
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.utils import (
    check_input_tensors_sequential,
    get_depth_dict,
    get_inner_layers,
)
from decomon.utils import Slope

OutputMapKey = Union[str, int]
OutputMapVal = Union[List[tf.Tensor], "OutputMapDict"]
OutputMapDict = Dict[OutputMapKey, OutputMapVal]

LayerMapVal = Union[List[DecomonLayer], "LayerMapDict"]
LayerMapDict = Dict[int, LayerMapVal]


def include_dim_layer_fn(
    layer_fn: Callable[..., List[DecomonLayer]],
    input_dim: int,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    IBP: bool = True,
    forward: bool = True,
    finetune: bool = False,
    shared: bool = True,
) -> Callable[[Layer], List[DecomonLayer]]:
    """include external parameters inside the translation of a layer to its decomon counterpart

    Args:
        layer_fn
        input_dim
        dc_decomp
        convex_domain
        finetune

    Returns:

    """
    if convex_domain is None:
        convex_domain = {}
    if "input_dim" in inspect.signature(layer_fn).parameters:
        layer_fn_copy = deepcopy(layer_fn)

        def func(layer: Layer) -> List[DecomonLayer]:
            return layer_fn_copy(
                layer,
                input_dim=input_dim,
                slope=slope,
                convex_domain=convex_domain,
                dc_decomp=dc_decomp,
                IBP=IBP,
                forward=forward,
                finetune=finetune,
                shared=shared,
            )

        layer_fn = func

    else:

        def func(layer: Layer) -> List[DecomonLayer]:
            return layer_fn(layer)

        layer_fn = func

    return layer_fn


def convert_forward(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]] = None,
    layer_fn: Callable[..., List[DecomonLayer]] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    input_dim: int = -1,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    IBP: bool = True,
    forward: bool = True,
    finetune: bool = False,
    shared: bool = True,
    softmax_to_linear: bool = True,
    joint: bool = True,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], LayerMapDict, OutputMapDict]:

    if convex_domain is None:
        convex_domain = {}
    if not isinstance(model, Model):
        raise ValueError()

    f_output = convert_forward_functional_model(
        model,
        input_tensors,
        layer_fn,
        slope=slope,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
        shared=shared,
        softmax_to_linear=softmax_to_linear,
        joint=joint,
        **kwargs,
    )

    return f_output


def convert_forward_functional_model(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]] = None,
    layer_fn: Callable[..., List[DecomonLayer]] = to_decomon,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    input_dim: int = 1,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    IBP: bool = True,
    forward: bool = True,
    finetune: bool = False,
    shared: bool = True,
    softmax_to_linear: bool = True,
    count: int = 0,
    joint: bool = True,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], LayerMapDict, OutputMapDict]:

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    input_dim_init: int
    if input_dim == -1:
        input_dim_init = -1
        if isinstance(model.input_shape, list):
            input_dim = np.prod(model.input_shape[0][1:])
        else:
            input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    if input_tensors is None:
        # check that the model has one input else
        input_tensors = []
        for i in range(len(model._input_layers)):

            tmp = check_input_tensors_sequential(model, None, input_dim, input_dim, IBP, forward, False, convex_domain)
            input_tensors += tmp

    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)

    has_iter = False
    if layer_fn is not None and len(layer_fn.__code__.co_varnames) == 1 and "layer" in layer_fn.__code__.co_varnames:
        has_iter = True

    if not has_iter:
        layer_fn = include_dim_layer_fn(
            layer_fn,
            input_dim=input_dim,
            slope=slope,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            shared=shared,
        )  # return a list of Decomon layers

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    # create input tensors

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    output_map: OutputMapDict = {}
    layer_map: LayerMapDict = {}
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
                _, output, layer_map_, output_map_ = convert_forward_functional_model(
                    model=layer,
                    input_tensors=output,
                    layer_fn=layer_fn,
                    input_dim=input_dim_init,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    finetune=finetune,
                    shared=shared,
                    softmax_to_linear=softmax_to_linear,
                    count=count,
                    **kwargs,
                )
                count = count + get_inner_layers(layer)
                layer_map[id(node)] = layer_map_
                output_map[id(node)] = output_map_
            else:

                list_layer_decomon = layer_fn(layer)
                layer_list: List[DecomonLayer] = []
                for layer_decomon in list_layer_decomon:
                    layer_decomon._name = f"{layer_decomon.name}_{count}"
                    count += 1
                    output = layer_decomon(output)
                    layer_list.append(layer_decomon)
                    if len(list_layer_decomon) > 1:
                        output_map[f"{id(node)}_{layer_decomon.name}"] = output
                layer_map[id(node)] = layer_list
            output_map[id(node)] = output

    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.layer.name for tensor in to_list(model.output)]
    for output_name in output_names:
        for node in output_nodes:
            if node.outbound_layer.name == output_name:
                output += output_map[id(node)]

    return input_tensors, output, layer_map, output_map
