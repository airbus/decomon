"""Module for MonotonicSequential.

It inherits from keras Sequential class.

"""
from __future__ import absolute_import
import inspect
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Flatten
from ..layers.decomon_layers import to_monotonic
from ..layers.core import Box, StaticVariables
from tensorflow.keras.layers import InputLayer, Input, Layer
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list
from copy import deepcopy
import numpy as np
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.layers.utils import get_upper, get_lower, linear_to_softmax
from ..backward_layers.backward_layers import get_backward as get_backward_
from ..backward_layers.backward_layers import join
from ..backward_layers.utils import backward_linear_prod
from ..backward_layers.utils import V_slope, S_slope

from .models import DecomonModel, DecomonSequential, Forward, Backward
from ..backward_layers.backward_layers import get_backward as get_backward_

from ..utils import M_BACKWARD, M_FORWARD, M_REC_BACKWARD
from .utils import (
    check_input_tensors_functionnal,
    check_input_tensors_sequential,
    include_dim_layer_fn,
    get_node_by_id,
    set_name,
    get_inputs,
    get_original_layer_name,
)


def get_forward_map(inputs, model, id_node_init=None):

    forward_map = {}
    list_layers = [l.name for l in model.layers]
    # for input_layer in model._input_layers:
    #    forward_map[input_layer.name]=inputs

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            id_node = get_node_by_id(node)
            layer = node.outbound_layer
            extra_names = get_original_layer_name(layer)

            if "{}_{}".format(layer.name, id_node) in forward_map:
                import pdb

                pdb.set_trace()

            if depth == max(depth_keys):
                if isinstance(layer, InputLayer):
                    forward_map["{}_{}".format(layer.name, id_node)] = inputs
                    for extra_name in extra_names:
                        forward_map["{}_{}".format(extra_name, id_node)] = inputs
                else:
                    tmp_0 = layer(inputs)
                    forward_map["{}_{}".format(layer.name, id_node)] = tmp_0
                    for extra_name in extra_names:
                        forward_map["{}_{}".format(extra_name, id_node)] = tmp_0
            else:
                node_inbound = to_list(node.parent_nodes)
                # input_layers = to_list(node.inbound_layers)
                inputs_ = []
                input_layer_only = True
                for node_i in node_inbound:
                    layer_i = node_i.outbound_layer
                    if not isinstance(layer_i, InputLayer):
                        input_layer_only = False
                    if layer_i.name in list_layers:
                        inputs_ += forward_map["{}_{}".format(layer_i.name, get_node_by_id(node_i))]
                if len(inputs_) == 0:
                    inputs_ = inputs

                # check whether the layer is connected to nothing or input layers

                if isinstance(layer, Model):
                    f_map = get_forward_map(inputs_, layer)
                    forward_map = {**forward_map, **f_map}

                tmp_1 = layer(inputs_)
                forward_map["{}_{}".format(layer.name, id_node)] = tmp_1
                for extra_name in extra_names:
                    forward_map["{}_{}".format(extra_name, id_node)] = tmp_1

                if input_layer_only and not (id_node_init is None):
                    forward_map["{}_{}".format(layer.name, id_node_init)] = tmp_1
                    for extra_name in extra_names:
                        forward_map["{}_{}".format(extra_name, id_node_init)] = tmp_1

    output = []
    nodes = model._nodes_by_depth[0]
    for node in nodes:
        id_node = get_node_by_id(node)
        layer = node.outbound_layer
        output += forward_map["{}_{}".format(layer.name, id_node)]

    return output, forward_map


def convert_forward(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=-1,
    dc_decomp=False,
    convex_domain={},
    IBP=True,
    forward=True,
    finetune=False,
    shared=True,
    softmax_to_linear=True,
):

    if not isinstance(model, Model):
        raise ValueError()
    """
    if isinstance(model, Sequential):
        f_output = convert_forward_sequential_model(
            model,
            input_tensors,
            layer_fn,
            input_dim=input_dim,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            shared=shared,
            softmax_to_linear=softmax_to_linear
        )
    else:
    """
    f_output = convert_forward_functional_model(
        model,
        input_tensors,
        layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
        shared=shared,
        softmax_to_linear=True,
    )

    return f_output


def convert_forward_functional_model(
    model,
    input_tensors=None,
    layer_fn=to_monotonic,
    input_dim=1,
    dc_decomp=False,
    convex_domain={},
    IBP=True,
    forward=True,
    finetune=False,
    shared=True,
    softmax_to_linear=True,
    layer_map={},
    forward_map={},
    name_history=set(),
):

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    if input_dim == -1:
        input_dim_init = -1
        input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    if input_tensors is None:
        input_tensors = check_input_tensors_sequential(
            model, input_tensors, input_dim, input_dim_init, IBP, forward, dc_decomp, convex_domain
        )

    has_softmax = False
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

    layer_fn = include_dim_layer_fn(
        layer_fn,
        input_dim=input_dim,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        IBP=IBP,
        forward=forward,
        finetune=finetune,
        shared=shared,
    )  # return a list of Decomon layers

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    # sort by depth
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    tensor_map = {}

    for depth in depth_keys:

        nodes_depth = model._nodes_by_depth[depth]
        for node in nodes_depth:

            id_node = get_node_by_id(node)
            layer_ = node.outbound_layer
            if "{}_{}".format(layer_, id_node) in tensor_map.keys():
                continue

            input_layers = to_list(node.inbound_layers)

            if isinstance(layer_, InputLayer):
                output = input_tensors
                tensor_map["{}_{}".format(layer_.name, id_node)] = output
                forward_map["{}_{}".format(layer_.name, id_node)] = output
                continue
            if len(input_layers) == 0:
                output = input_tensors
            else:

                output = get_inputs(node, tensor_map)
            if isinstance(layer_, Model):
                input_layer_, output_layer_, l_map, _ = convert_forward_functional_model(
                    layer_,
                    input_dim=input_dim,
                    layer_fn=layer_fn,
                    dc_decomp=dc_decomp,
                    shared=shared,
                    finetune=finetune,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    name_history=name_history,
                )
                name_ = "{}_to_monotonic".format(layer_.name)
                """
                if '{}_to_monotonic'.format(layer_.name) in name_history:
                    count=0
                    while '{}_to_monotonic_{}'.format(layer_.name, count) in name_history:
                        count+=1
                    name_ = '{}_to_monotonic_{}'.format(layer_.name, count)
                name_history.add(name_)
                """

                layer_decomon = [Model(input_layer_, output_layer_, name=name_)]
            else:
                layer_decomon = layer_fn(layer_)

            # rename if necessary
            for layer_decomon_i in layer_decomon:
                if layer_decomon_i.name in name_history:
                    count = 0
                    while "{}_{}".format(layer_decomon_i.name, count) in name_history:
                        count += 1
                    # set the name in l_map as well
                    set_name(layer_decomon_i, count)
                name_history.add(layer_decomon_i.name)

            if isinstance(layer_, Model):
                input_model = get_inputs(node, tensor_map)

                # tmp_model = Model(input_model, layer_decomon[0](input_model))
                output, f_map = get_forward_map(input_model, layer_decomon[0])

                # f_map = get_forward_map(input_model, tmp_model)
                forward_map = {**forward_map, **f_map}
            else:

                for layer_decomon_i in layer_decomon:
                    output = layer_decomon_i(output)
                    # forward_map[layer_decomon_i.name] = output

                    forward_map["{}_{}".format(layer_decomon_i.name, id_node)] = output

            forward_map["{}_{}".format(layer_.name, id_node)] = output
            tensor_map["{}_{}".format(layer_.name, id_node)] = output

            if isinstance(layer_, Model):
                # layer_map = {**layer_map, **l_map}
                layer_map["{}_{}".format(layer_.name, id_node)] = l_map
            else:
                layer_map["{}_{}".format(layer_.name, id_node)] = layer_decomon

    output = []
    nodes = model._nodes_by_depth[0]
    for node in nodes:
        id_node = get_node_by_id(node, True)
        output += tensor_map[id_node]
    return input_tensors, output, layer_map, forward_map
