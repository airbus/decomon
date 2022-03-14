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
    pre_process_inputs,
    get_mode,
)


def get_forward_map(inputs, model, id_node_init=None):

    forward_map = {}
    list_layers = [l.name for l in model.layers]
    # for input_layer in model._input_layers:
    #    forward_map[input_layer.name]=inputs

    depth_keys = list(model._nodes_by_depth.keys())

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


def get_forward_mapping_model(model, layer_map, input_tensors, mode):

    f_map = {}
    tensor_map = {}
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # step 1: init input_tensors:
    input_tensors_ = []
    nodes_inits = model._nodes_by_depth[depth_keys[0]]
    nb_comp = int(len(input_tensors) / len(nodes_inits))
    input_tensors_ = [input_tensors[i * nb_comp : (i + 1) * nb_comp] for i in range(len(nodes_inits))]

    for node_init, input_i in zip(nodes_inits, input_tensors_):
        layer_ = node_init.outbound_layer
        layer_name = layer_.name
        id_node = get_node_by_id(node_init)

        if isinstance(layer_, Model):
            output_layer_, f_map_from_root = get_forward_mapping_model(
                layer_, layer_map["{}_{}".format(layer_.name, id_node)], input_i, mode
            )
            output_from_input_layer = pre_process_inputs(input_i, mode)
            output_disconnected, f_map_disconnected = get_forward_mapping_model(
                layer_, layer_map["{}_{}".format(layer_.name, id_node)], output_from_input_layer, mode
            )

            tensor_map["{}_{}".format(layer_.name, id_node)] = output_layer_
            f_map["{}_{}".format(layer_name, id_node)] = [
                output_layer_,
                f_map_from_root,
                output_disconnected,
                f_map_disconnected,
            ]
        else:
            decomon_inputs = layer_map["{}_{}".format(layer_name, id_node)]
            output_i = decomon_inputs[0](input_i)
            f_map["{}_{}".format(decomon_inputs[0].name, id_node)] = output_i
            if len(decomon_inputs) > 1:
                for k in range(1, len(decomon_inputs)):
                    output_i = decomon_inputs[1](output_i)
                    f_map["{}_{}".format(decomon_inputs[k].name, id_node)] = output_i
            tensor_map["{}_{}".format(layer_name, id_node)] = output_i

    for depth in depth_keys[1:]:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            layer_ = node.outbound_layer
            id_node = get_node_by_id(node)

            inputs_ = []
            for node_in in node.parent_nodes:
                if "{}_{}".format(node_in.outbound_layer.name, get_node_by_id(node_in)) not in tensor_map:
                    import pdb

                    pdb.set_trace()
                inputs_ += tensor_map["{}_{}".format(node_in.outbound_layer.name, get_node_by_id(node_in))]

            if isinstance(layer_, Model):
                output_layer_, f_map_from_root = get_forward_mapping_model(
                    layer_, layer_map["{}_{}".format(layer_.name, id_node)], inputs_, mode
                )
                output_from_input_layer = pre_process_inputs(inputs_, mode)
                output_disconnected, f_map_disconnected = get_forward_mapping_model(
                    layer_, layer_map["{}_{}".format(layer_.name, id_node)], output_from_input_layer, mode
                )

                tensor_map["{}_{}".format(layer_.name, id_node)] = output_layer_
                f_map["{}_{}".format(layer_name, id_node)] = [
                    output_layer_,
                    f_map_from_root,
                    output_disconnected,
                    f_map_disconnected,
                ]

            else:
                # retrieve inputs

                decomon_layer_ = layer_map["{}_{}".format(layer_.name, id_node)]
                output_i = decomon_layer_[0](inputs_)
                f_map["{}_{}".format(decomon_layer_[0].name, id_node)] = output_i

                if len(decomon_layer_) > 1:
                    for k in range(1, len(decomon_layer_)):
                        output_i = decomon_layer_[k](output_i)
                        f_map["{}_{}".format(decomon_layer_[k].name, id_node)] = output_i
                tensor_map["{}_{}".format(layer_.name, id_node)] = output_i

    # get output nodes
    nodes_output = model._nodes_by_depth[0]
    output = []
    for node_output in nodes_output:

        output += tensor_map["{}_{}".format(node_output.outbound_layer.name, get_node_by_id(node_output))]

    return output, f_map


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
    **kwargs,
):

    if not isinstance(model, Model):
        raise ValueError()

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
    finetune_output=False,
    opt_linear=True, # detect linear successive parts by design
):

    if not isinstance(model, Model):
        raise ValueError("Expected `model` argument " "to be a `Model` instance, got ", model)

    if input_dim == -1:
        input_dim_init = -1
        if isinstance(model.input_shape, list):
            input_dim = np.prod(model.input_shape[0][1:])
        else:
            input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    input_tensors_dict = {}
    if input_tensors is None:
        # check that the model has one input else
        input_tensors = []
        for i in range(len(model._input_layers)):

            tmp = check_input_tensors_sequential(
                model, None, input_dim, input_dim_init, IBP, forward, dc_decomp, convex_domain
            )
            input_tensors_dict[model._input_layers[i].name] = tmp
            input_tensors += tmp
    else:
        nb_tensors = 0
        if IBP and forward:
            nb_tensors = 7
        if not IBP and forward:
            nb_tensors = 5
        if IBP and not forward:
            nb_tensors = 2

        if len(input_tensors) == nb_tensors:
            for layer in model._input_layers:
                input_tensors_dict[layer.name] = input_tensors
        else:
            order_name = [node.outbound_layer.name for node in model._nodes_by_depth[max(depth_keys)]]
            for i in range(len(model._input_layers)):
                layer = model._input_layers[i]
                j = np.argmax([elem == layer.name for elem in order_name])
                input_tensors_dict[layer.name] = input_tensors[j * nb_tensors : (j + 1) * nb_tensors]

    has_softmax = False
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

    has_iter = False
    if (
        layer_fn is not None
        and len(layer_fn.__code__.co_varnames) == 2
        and "layer" in layer_fn.__code__.co_varnames
        and "depth" in layer_fn.__code__.co_varnames
    ):
        has_iter = True

    if not has_iter:
        layer_fn_ = include_dim_layer_fn(
            layer_fn,
            input_dim=input_dim,
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            IBP=IBP,
            forward=forward,
            finetune=finetune,
            shared=shared,
        )  # return a list of Decomon layers

        if not finetune_output and finetune:

            layer_fn_output = include_dim_layer_fn(
                layer_fn,
                input_dim=input_dim,
                dc_decomp=dc_decomp,
                convex_domain=convex_domain,
                IBP=IBP,
                forward=forward,
                finetune=False,
                shared=shared,
            )

            def func(layer, depth):
                if depth:
                    return layer_fn_(layer)
                else:
                    return layer_fn_output(layer)

        else:

            def func(layer, depth):
                return layer_fn_(layer)

        layer_fn = func

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

    # sort by depth

    # start with inputs layers

    tensor_map = {}

    for count_, depth in enumerate(depth_keys):

        nodes_depth = model._nodes_by_depth[depth]
        for node in nodes_depth:

            id_node = get_node_by_id(node)
            layer_ = node.outbound_layer
            if "{}_{}".format(layer_, id_node) in tensor_map.keys():
                continue  # to check

            input_layers = to_list(node.inbound_layers)

            if len(input_layers) == 0:
                output = input_tensors_dict[layer_.name]
                # output = input_tensors
            else:
                if depth == max(depth_keys) and layer_.name in input_tensors_dict.keys():
                    output = input_tensors_dict[layer_.name]
                else:
                    output = get_inputs(node, tensor_map)

            if isinstance(layer_, Model):
                opt_linear_ = False
                if count_==0 and opt_linear:
                    opt_linear_=True
                if opt_linear:
                    # check the state of the previous layers
                    raise NotImplementedError()
                toto, output_, l_map, f_map = convert_forward_functional_model(
                    layer_,
                    input_tensors=output,
                    input_dim=input_dim,
                    layer_fn=layer_fn,
                    dc_decomp=dc_decomp,
                    shared=shared,
                    finetune=finetune,
                    convex_domain=convex_domain,
                    IBP=IBP,
                    forward=forward,
                    name_history=name_history,
                    opt_init=opt_linear_
                )

                # output_from_input = pre_process_inputs(output, get_mode(IBP, forward))
                output = output_
                # output_tmp, f_map_tmp = get_forward_mapping_model(layer_, l_map, output_from_input, get_mode(IBP, forward))

                # update name_history...
                # update name_history
                for key_model in l_map.keys():
                    internal_layer_name = key_model.split("_NODE_")[0]
                    name_history.add(internal_layer_name)

            else:
                layer_decomon = layer_fn(layer_, depth)
                if count_==0:
                    layer_decomon[0].set_linear(opt_linear)
                if opt_linear and count_:
                    parents = to_list(node.parent_nodes)

                    bool_linear = min([layer_map["{}_{}".format(parent.outbound_layer.name, get_node_by_id(parent))][-1].get_linear() for parent in parents])
                    layer_decomon[0].set_linear(bool_linear)
                # rename if necessary
                for layer_decomon_i in layer_decomon:
                    if layer_decomon_i.name in name_history:
                        count = 0
                        while "{}_{}".format(layer_decomon_i.name, count) in name_history:
                            count += 1
                        # set the name in l_map as well
                        set_name(layer_decomon_i, count)
                    name_history.add(layer_decomon_i.name)
                    output = layer_decomon_i(output)
                    forward_map["{}_{}".format(layer_decomon_i.name, id_node)] = output

            tensor_map["{}_{}".format(layer_.name, id_node)] = output

            if isinstance(layer_, Model):
                # layer_map = {**layer_map, **l_map}
                layer_map["{}_{}".format(layer_.name, id_node)] = l_map
                forward_map["{}_{}".format(layer_.name, id_node)] = [output, f_map]  # , output_tmp, f_map_tmp]
            else:
                layer_map["{}_{}".format(layer_.name, id_node)] = layer_decomon
                forward_map["{}_{}".format(layer_.name, id_node)] = output

    output_list = [[] for _ in range(len(model._output_layers))]
    order_name = [layer.name for layer in model._output_layers]

    nodes = model._nodes_by_depth[0]

    for node in nodes:
        id_node = get_node_by_id(node, True)
        output_node = tensor_map[id_node]

        index = np.argmax([elem == node.outbound_layer.name for elem in order_name])
        output_list[index] = output_node

    output = []
    for elem in output_list:
        output += elem

    if has_softmax:
        linear_to_softmax(model)

    return input_tensors, output, layer_map, forward_map
