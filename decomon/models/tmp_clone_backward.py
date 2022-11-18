from __future__ import absolute_import

import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import (
    Concatenate,
    Flatten,
    Input,
    InputLayer,
    Lambda,
    Layer,
    Merge,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.layers.decomon_layers import DecomonMaximum, DecomonMinimum

from ..backward_layers.backward_layers import get_backward as get_backward_
from ..utils import (
    F_FORWARD,
    F_HYBRID,
    F_IBP,
    backward_maximum,
    backward_minimum,
    get_lower_layer,
    get_upper_layer,
)
from .utils import (
    check_input_tensors_sequential,
    convert_to_backward_bounds,
    fuse_forward_backward,
    get_back_bounds_model,
    get_FORWARD,
    get_IBP,
    get_input_dim,
    get_key,
    get_mode,
    get_node_by_id,
)


def update_input(backward_bound, input_tensors, mode, output_shape, **kwargs):

    op_reshape_u = Reshape(output_shape)
    if mode in [F_HYBRID.name, F_FORWARD.name]:
        op_reshape_w = Reshape([-1] + to_list(output_shape))

    if mode == F_HYBRID.name:
        x, u_c, w_u, b_u, l_c, w_l, b_l = input_tensors
    if mode == F_FORWARD.name:
        x, w_u, b_u, w_l, b_l = input_tensors
    if mode == F_IBP.name:
        u_c, l_c = input_tensors

        if "flatten" in kwargs:
            op_flatten = kwargs["flatten"]
        else:
            op_flatten = Flatten()

        if "reshape" in kwargs:
            op_reshape = kwargs["reshape"]
        else:
            op_reshape = Reshape((1, -1))

        u_c_flat = op_reshape(op_flatten(u_c))
        l_c_flat = op_reshape(op_flatten(l_c))

        if "concatenate" in kwargs:
            op_concat = kwargs["concatenate"]
        else:
            op_concat = Concatenate(axis=1)

        if "upper_layer" in kwargs:
            upper_layer = kwargs["upper_layer"]
            lower_layer = kwargs["lower_layer"]
        else:
            if "convex_domain" in kwargs:
                convex_domain = kwargs["convex_domain"]
            else:
                convex_domain = {}

            upper_layer = get_upper_layer(convex_domain)
            lower_layer = get_lower_layer(convex_domain)

        x = op_concat([l_c_flat, u_c_flat])
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bound
        u_c_out = op_reshape_u(upper_layer(x, w_out_u, b_out_u))
        l_c_out = op_reshape_u(lower_layer(x, w_out_l, b_out_l))

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        # update the linear bounds

        def func(variables):

            w_u, b_u, w_l, b_l, w_out_u, b_out_u, w_out_l, b_out_l = variables
            if len(w_u.shape) == 2:
                return w_out_u, b_out_u, w_out_l, b_out_l
            z_value = 0.0
            w_u_ = K.reshape(w_u, [-1, w_u.shape[1], np.prod(w_u.shape[2:])])
            w_l_ = K.reshape(w_l, [-1, w_l.shape[1], np.prod(w_l.shape[2:])])
            b_u_ = K.reshape(b_u, [-1, np.prod(b_u.shape[1:])])
            b_l_ = K.reshape(b_l, [-1, np.prod(b_l.shape[1:])])

            w_u_pos = K.maximum(w_out_u, z_value)
            w_u_neg = K.minimum(w_out_u, z_value)
            w_l_pos = K.maximum(w_out_l, z_value)
            w_l_neg = K.minimum(w_out_l, z_value)

            w_out_u_ = K.sum(K.expand_dims(w_u_pos, 1) * K.expand_dims(w_u_, -1), 2) + K.sum(
                K.expand_dims(w_u_neg, 1) * K.expand_dims(w_l_, -1), 2
            )
            w_out_l_ = K.sum(K.expand_dims(w_l_pos, 1) * K.expand_dims(w_l_, -1), 2) + K.sum(
                K.expand_dims(w_l_neg, 1) * K.expand_dims(w_u_, -1), 2
            )

            b_out_u_ = (
                K.sum(w_u_pos * K.expand_dims(b_u_, -1), 1) + K.sum(w_u_neg * K.expand_dims(b_l_, -1), 1) + b_out_u
            )
            b_out_l_ = (
                K.sum(w_l_pos * K.expand_dims(b_l_, -1), 1) + K.sum(w_l_neg * K.expand_dims(b_u_, -1), 1) + b_out_l
            )

            return w_out_u_, b_out_u_, w_out_l_, b_out_l_

        lambda_ = Lambda(func)

        w_out_u_, b_out_u_, w_out_l_, b_out_l_ = lambda_([w_u, b_u, w_l, b_l] + backward_bound)
        w_out_u_ = op_reshape_w(w_out_u_)
        w_out_l_ = op_reshape_w(w_out_l_)
        b_out_u_ = op_reshape_u(b_out_u_)
        b_out_l_ = op_reshape_u(b_out_l_)
        # need a reshape at the end

    if mode == F_HYBRID.name:
        if "upper_layer" in kwargs:
            upper_layer = kwargs["upper_layer"]
            lower_layer = kwargs["lower_layer"]
        else:
            if "convex_domain" in kwargs:
                convex_domain = kwargs["convex_domain"]
            else:
                convex_domain = {}

            upper_layer = get_upper_layer(convex_domain)
            lower_layer = get_lower_layer(convex_domain)
        u_c_out = upper_layer(x, w_out_u_, b_out_u_)
        l_c_out = lower_layer(x, w_out_l_, b_out_l_)

    if mode == F_IBP.name:
        return [u_c_out, l_c_out]
    if mode == F_FORWARD.name:
        return [x, w_out_u_, b_out_u_, w_out_l_, b_out_l_]
    if mode == F_HYBRID.name:
        return [x, u_c_out, w_out_u_, b_out_u_, l_c_out, w_out_l_, b_out_l_]


def fuse_backward_bounds(back_bounds_list, input_tensors, mode, **kwargs):

    if len(back_bounds_list) == 1:
        return back_bounds_list[0]

    if mode == F_IBP.name:
        u_c, l_c = input_tensors
    if mode == F_HYBRID.name:
        _, u_c, _, _, l_c, _, _ = input_tensors
    if mode == F_FORWARD.name:
        # check upper and lower function in kwargs
        if "upper_layer" in kwargs:
            upper_layer = kwargs["upper_layer"]
            lower_layer = kwargs["lower_layer"]
        else:
            if "convex_domain" in kwargs:
                convex_domain = kwargs["convex_domain"]
            else:
                convex_domain = {}

            upper_layer = get_upper_layer(convex_domain)
            lower_layer = get_lower_layer(convex_domain)

        # compute u_c and l_c
        x, w_u, b_u, w_l, b_l = input_tensors
        u_c = upper_layer([x, w_u, b_u])
        l_c = lower_layer([x, w_l, b_l])

    if "flatten" in kwargs:
        op_flatten = kwargs["flatten"]
    else:
        op_flatten = Flatten()

    if "reshape" in kwargs:
        op_reshape = kwargs["reshape"]
    else:
        op_reshape = Reshape((1, -1))

    u_c_flat = op_reshape(op_flatten(u_c))
    l_c_flat = op_reshape(op_flatten(l_c))

    if "concatenate" in kwargs:
        op_concat = kwargs["concatenate"]
    else:
        op_concat = Concatenate(axis=1)
    x_ = op_concat([l_c_flat, u_c_flat])

    inputs_ = []
    for bound_i in back_bounds_list:
        inputs_ += [x_] + bound_i

    max_bounds = DecomonMinimum(mode=F_FORWARD.name)(inputs_)
    min_bounds = DecomonMaximum(mode=F_FORWARD.name)(inputs_)

    def func(inputs_):
        n = int(len(inputs_) / 2)
        inputs_0 = inputs_[:n]
        inputs_1 = inputs_[n:]
        x, w_u, b_u = inputs_0[:3]
        w_l, b_l = inputs_1[-2:]

        return [w_u, b_u, w_l, b_l]

    lambda_f = Lambda(func)
    output = lambda_f(max_bounds + min_bounds)
    return output


def get_backward_layer_input(layer_, id_node, forward_map, mode, back_bounds, finetune, convex_domain):

    # assume back_bounds is either an empty dict

    layer_back = get_backward_(
        layer_, previous=(len(back_bounds) > 0), mode=mode, finetune=finetune, convex_domain=convex_domain
    )
    input_layer_ = forward_map[f"{layer_.name}_{id_node}"]
    back_bounds_ = layer_back(input_layer_ + back_bounds)
    if isinstance(back_bounds_, tuple):
        back_bounds_ = list(back_bounds_)

    return {f"{layer_.name}_{id_node}": back_bounds}


def get_output_layer(node, layer_map, forward_map, mode, **kwargs):

    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)
    if f"{layer_.name}_{id_node}" in forward_map:
        return forward_map[f"{layer_.name}_{id_node}"]
    else:
        if len(layer_.output_shape) > 1:
            raise NotImplementedError("Decomon cannot handle nodes with a list of output tensors")
        backward_map = get_backward_layer(node, layer_map, forward_map, mode, [], **kwargs)
        outputs = []
        for key in backward_map:
            outputs += update_input(backward_map[key], forward_map[key], mode, layer_.output_shape, **kwargs)

        if "convex_domain" in kwargs:
            convex_domain = kwargs["convex_domain"]
        else:
            convex_domain = {}
        output_max = DecomonMinimum(mode=mode, convex_domain=convex_domain)(outputs)
        output_min = DecomonMaximum(mode=mode, convex_domain=convex_domain)(outputs)

        if mode == F_IBP.name:
            return [output_max[0], output_min[-1]]
        if mode == F_FORWARD.name:
            return output_max[:3] + output_min[-2:]
        if mode == F_HYBRID.name:
            return output_max[:4] + output_min[-3:]


def get_backward_layer(node, layer_map, forward_map, mode, back_bounds, **kwargs):

    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)
    finetune = False
    if "finetune" in kwargs:
        finetune = kwargs["finetune"]
    convex_domain = {}
    if "convex_domain" in kwargs:
        convex_domain = kwargs["convex_domain"]
    f_map = {}
    l_map = {}

    if isinstance(layer_, InputLayer):
        # return back_bounds, dict of backward bounds associated to inputs
        return get_backward_layer_input(layer_, id_node, forward_map, mode, back_bounds, finetune, convex_domain)

    if f"{layer_.name}_{id_node}" not in layer_map:
        layer_list = [layer_]
    else:
        layer_list = to_list(layer_map[f"{layer_.name}_{id_node}"])
        layer_list = layer_list[::-1]

    for i in range(len(layer_list) - 1):
        # if layer_list has more than one element, then there has been a forward pass
        # hence forward_map contains the related input
        inputs_i = forward_map[f"{layer_list[i + 1].name}_{id_node}"]
        back_layer_i = get_backward_(
            layer_, previous=(len(back_bounds) > 0), mode=mode, finetune=finetune, convex_domain=convex_domain
        )
        back_bounds_ = back_layer_i(inputs_i + back_layer_i)
        if isinstance(back_bounds_, tuple):
            back_bounds_ = list(back_bounds_)
        back_bounds = back_bounds_

    layer_ = layer_list[-1]
    input_nodes = to_list(node.parent_nodes)
    # get input nodes
    input_layer_ = [None] * len(layer_.input_shape)
    if isinstance(layer_, Model):
        if len(to_list(layer_.output_shape)) > 1:
            raise NotImplementedError("Decomon cannot handle nodes with a list of output tensors")

        finetune_ = False
        convex_domain_ = {}

        if "finetune" in kwargs:
            finetune_ = kwargs["finetune"]
        if "convex_domain" in kwargs:
            convex_domain_ = kwargs["convex_domain"]

        back_bounds_ = get_backward_model(
            layer_,
            input_tensors=input_layer_,
            back_bounds=back_bounds,
            convex_domain=convex_domain_,
            IBP=get_IBP(mode),
            forward=get_FORWARD(mode),
            finetune=finetune_,
            fuse_with_input=False,
            **kwargs,
        )
        if len(layer_.input_shape) > 1:
            back_bounds_ = back_bounds_[0]

    else:
        back_layer = get_backward_(
            layer_, previous=(len(back_bounds) > 0), mode=mode, finetune=finetune, convex_domain=convex_domain
        )
        back_bounds_ = back_layer(inputs_i + back_layer_i)
    if isinstance(back_bounds_, tuple):
        back_bounds_ = list(back_bounds_)
    back_bounds = back_bounds_

    if len(input_nodes) == 1:
        return get_backward_layer(input_nodes[0], layer_map, forward_map, mode, back_bounds, **kwargs)
    else:
        dico_backward = {}
        for node_i, bound_i in zip(input_nodes, back_bounds):
            backward_map_i = get_backward_layer(node_i, layer_map, forward_map, mode, bound_i, **kwargs)
            for key_i in backward_map_i:
                if key_i not in dico_backward:
                    dico_backward[key_i] = [backward_map_i[key_i]]
                else:
                    dico_backward[key_i] += backward_map_i[key_i]

        # fuse several elements if needed
        for key in dico_backward:
            dico_backward[key] = fuse_backward_bounds(dico_backward[key], forward_map[key], mode, **kwargs)
        return dico_backward


def get_backward_model(
    model,
    input_tensors=None,
    back_bounds=None,
    input_dim=-1,
    convex_domain=None,
    IBP=True,
    forward=True,
    finetune=False,
    layer_map=None,
    forward_map=None,
    fuse_with_input=True,
    **kwargs,
):
    if back_bounds is None:
        back_bounds = []
    if convex_domain is None:
        convex_domain = {}
    if layer_map is None:
        layer_map = {}
    if forward_map is None:
        forward_map = {}
    if not isinstance(model, Model):
        raise ValueError()

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    nb_output = len(to_list(model.output_shape))
    nb_input = len(to_list(model.input_shape))
    nodes_input = model._nodes_by_depth[max(depth_keys)]

    if input_dim == -1:
        input_dim_init = -1
        input_dim = np.prod(model.input_shape[1:])
    else:
        input_dim_init = input_dim

    if input_tensors is None:
        # check that the model has one input else
        for node_i in nodes_input:
            layer_i = node_i.outbound_layer
            id_node_i = get_node_by_id(node_i)
            if f"{layer_i}_{id_node_i}" not in forward_map:
                if input_dim == -1:
                    input_dim_i = np.prod(layer_i.input_shape[1:])
                else:
                    input_dim_i = input_dim

                forward_map[f"{layer_i}_{id_node_i}"] = check_input_tensors_sequential(
                    model, None, input_dim_i, input_dim_init, IBP, forward, False, convex_domain
                )
    else:
        for i, node_i in enumerate(nodes_input):
            layer_i = node_i.outbound_layer
            id_node_i = get_node_by_id(node_i)
            if f"{layer_i}_{id_node_i}" not in forward_map:
                forward_map[f"{layer_i}_{id_node_i}"] = input_tensors[i]
            else:
                raise ValueError

    mode = get_mode(IBP=IBP, forward=forward)
    assert isinstance(back_bounds, list)
    assert len(back_bounds) in [0, nb_output]
    if len(back_bounds) == 0:
        back_bounds = [[]] * nb_output

    nodes_output = model._nodes_by_depth[0]
    output = []
    for node_i, bounds_i in zip(nodes_output, back_bounds):
        back_bound_i_dict = get_backward_layer(
            node_i, layer_map, forward_map, mode=mode, back_bounds=to_list(bounds_i), **kwargs
        )
        back_bound_i = []

        if nb_input == 1:
            back_bound_i = back_bound_i_dict.values
        else:

            for node_j in nodes_input:
                layer_name_j = node_j.outbound_layer.name
                id_j = get_node_by_id(node_j)
                inputs_tensors_i = forward_map[f"{layer_name_j.name}_{id_j}"]
                if f"{layer_name_j}_{id_j}" in back_bound_i_dict:
                    output_i = back_bound_i_dict[f"{layer_name_j}_{id_j}"]
                    if mode == F_IBP.name:
                        back_bound_i.append(output_i)
                    else:
                        if mode == F_HYBRID.name:
                            output_i_ = update_input(
                                output_i,
                                [inputs_tensors_i[k] for k in [0, 2, 3, 5, 6]],
                                F_FORWARD.name,
                                node_i.outbound_layer.output_shape,
                                **kwargs,
                            )[1:]
                        else:
                            output_i_ = update_input(
                                output_i, inputs_tensors_i, mode, node_i.outbound_layer.output_shape, **kwargs
                            )[1:]
                        back_bound_i.append(output_i)
                else:
                    # create fake input
                    raise NotImplementedError()

        output += back_bound_i

    # here update layer_map annd forward_map
    if nb_output == 1 and nb_input == 1:
        return output[0]

    return output
