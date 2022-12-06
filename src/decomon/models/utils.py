import inspect
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import (
    Concatenate,
    Flatten,
    Input,
    Lambda,
    Maximum,
    Minimum,
)
from tensorflow.keras.models import Model

from decomon.layers.core import F_FORWARD, F_HYBRID, F_IBP, Box
from decomon.utils import get_lower, get_lower_layer, get_upper, get_upper_layer


def include_dim_layer_fn(
    layer_fn, input_dim, dc_decomp=False, convex_domain=None, IBP=True, forward=True, finetune=False, shared=True
):
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
    if input_dim <= 0:
        raise ValueError()
    else:
        if "input_dim" in inspect.signature(layer_fn).parameters:
            layer_fn_copy = deepcopy(layer_fn)
            if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
                input_dim = (2, input_dim)
            else:
                if convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
                    input_dim = (2, input_dim)

            def func(x):
                return layer_fn_copy(
                    x,
                    input_dim,
                    convex_domain=convex_domain,
                    dc_decomp=dc_decomp,
                    IBP=IBP,
                    forward=forward,
                    finetune=finetune,
                    shared=shared,
                )

            layer_fn = func

        else:
            return layer_fn

    return layer_fn


def check_input_tensors_sequential(
    model, input_tensors, input_dim, input_dim_init, IBP, forward, dc_decomp, convex_domain
):

    if convex_domain is None:
        convex_domain = {}
    if input_tensors is None:  # no predefined input tensors

        input_shape = list(K.int_shape(model.input)[1:])

        if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
        elif convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
        else:

            if isinstance(input_dim, tuple):
                z_tensor = Input(input_dim, dtype=model.layers[0].dtype)
            else:
                z_tensor = Input((input_dim,), dtype=model.layers[0].dtype)

        if dc_decomp:
            h_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            g_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
        if forward:
            b_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            b_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if input_dim_init > 0:
                w_u_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
            else:
                w_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)

        if IBP:
            u_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            l_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if forward:  # hybrid mode
                input_tensors = [
                    z_tensor,
                    u_c_tensor,
                    w_u_tensor,
                    b_u_tensor,
                    l_c_tensor,
                    w_l_tensor,
                    b_l_tensor,
                ]
            else:
                # only IBP
                input_tensors = [
                    u_c_tensor,
                    l_c_tensor,
                ]
        elif forward:
            # forward mode
            input_tensors = [
                z_tensor,
                w_u_tensor,
                b_u_tensor,
                w_l_tensor,
                b_l_tensor,
            ]
        else:
            raise NotImplementedError("not IBP and not forward not implemented")

        if dc_decomp:
            input_tensors += [h_tensor, g_tensor]

    else:
        # assert that input_tensors is a List of 6 InputLayer objects
        # If input tensors are provided, the original model's InputLayer is
        # overwritten with a different InputLayer.
        assert isinstance(input_tensors, list), "expected input_tensors to be a List or None, but got dtype={}".format(
            input_tensors.dtype
        )

        if dc_decomp:
            if IBP and forward:
                assert len(input_tensors) == 9, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 4, "wrong number of inputs, expexted 6 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 8 but got {}".format(
                    len(input_tensors)
                )
        else:
            if IBP and forward:
                assert len(input_tensors) == 7, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if IBP and not forward:
                assert len(input_tensors) == 2, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if not IBP and forward:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )

    return input_tensors


def get_input_tensor_x(model, input_tensors, input_dim, input_dim_init, convex_domain):
    if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    elif convex_domain["name"] == Box.name and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    else:

        if isinstance(input_dim, tuple):
            z_tensor = Input(input_dim, dtype=model.layers[0].dtype)
        else:
            z_tensor = Input((input_dim,), dtype=model.layers[0].dtype)
    return z_tensor


def check_input_tensors_functionnal(
    model, input_tensors, input_dim, input_dim_init, IBP, forward, dc_decomp, convex_domain
):

    raise NotImplementedError()


def get_mode(IBP=True, forward=True):

    if IBP:
        if forward:
            return F_HYBRID.name
        else:
            return F_IBP.name
    else:
        return F_FORWARD.name


def get_IBP(mode=F_HYBRID.name):
    if mode in [F_HYBRID.name, F_IBP.name]:
        return True
    return False


def get_FORWARD(mode=F_HYBRID.name):
    if mode in [F_HYBRID.name, F_FORWARD.name]:
        return True
    return False


def get_node_by_id_(node):

    return f"NODE_{node.flat_output_ids}_{node.flat_input_ids}"


def get_node_by_id(node, outbound=False, model=None):
    layer_ = node.outbound_layer
    input_names = str(id(node))
    if outbound:
        return f"{layer_.name}_NODE_{input_names}"
    return f"NODE_{input_names}"


def set_name(layer, extra_id):

    layer._name = f"{layer.name}_{extra_id}"


def get_inputs(node, tensor_map):
    output = []
    # get parents nodes
    node_prev = node.parent_nodes
    if len(node_prev) == 0:
        raise ValueError()
    key_prev_id = [get_node_by_id(node_p_i, True) for node_p_i in node_prev]
    for key_p_i in key_prev_id:
        output += tensor_map[key_p_i]
    return output


def convert_to_backward_bounds(mode, inputs, input_dim):

    if mode == F_HYBRID.name:
        _, _, w_u, b_u, _, w_l, b_l = inputs
    elif mode == F_FORWARD.name:
        _, w_u, b_u, w_l, b_l = inputs
    elif mode == F_IBP.name:
        u_c, l_c = inputs
        z_value = K.cast(0.0, u_c.dtype)
        n_out = np.prod(u_c.shape[1:])
        b_u = K.reshape(u_c, (-1, n_out))
        b_l = K.reshape(l_c, (-1, n_out))
        w_u = K.zeros((1, input_dim, 1)) + z_value * K.expand_dims(b_u, 1)
        w_l = K.zeros((1, input_dim, 1)) + z_value * K.expand_dims(b_l, 1)
    else:
        raise ValueError(f"Unknown mode {mode}")
    return w_u, b_u, w_l, b_l


def get_input_dim(x_tensor):
    return x_tensor.shape[-1]


def get_original_layer_name(layer, pattern="_monotonic"):

    pattern_layer = layer.name.split(pattern)
    if len(pattern_layer) == 1:
        return []
    return ["".join(pattern_layer[:-1])]


def get_key(layer, id_node, forward_map):
    layer_name = layer.name
    n_word = len(layer_name)

    keys = [key for key in forward_map if key[:n_word] == layer_name]
    if len(keys) == 0:
        raise KeyError
    if len(keys) == 1:
        return keys[0]
    else:
        raise NotImplementedError()


def get_back_bounds_model(back_bounds, model):

    depth = 0  # start from the beginning
    nodes = model._nodes_by_depth[depth]
    back_bounds_list = []
    n = int(len(back_bounds) / len(nodes))
    for i in range(len(nodes)):
        back_bounds_list.append(back_bounds[i * n : (i + 1) * n])
    return back_bounds_list


def fuse_forward_backward(
    mode, inputs, back_bounds, upper_layer=None, lower_layer=None, convex_domain=None, x_tensor=None
):

    if convex_domain is None:
        convex_domain = {}
    if mode in [F_IBP.name, F_HYBRID.name]:

        if upper_layer is None:
            upper_layer = get_upper_layer(convex_domain)
        if lower_layer is None:
            lower_layer = get_lower_layer(convex_domain)

    if mode == F_IBP.name:
        w_out_u, b_out_u, w_out_l, b_out_l = back_bounds

        if x_tensor is None:
            raise ValueError  # we need some information on the domain
        u_out_c = upper_layer([x_tensor, w_out_u, b_out_u])
        l_out_c = lower_layer([x_tensor, w_out_l, b_out_l])

        return [u_out_c, l_out_c]
    elif mode in {F_FORWARD.name, F_HYBRID.name}:
        if mode == F_FORWARD.name:
            _, w_u, b_u, w_l, b_l = inputs
        else:
            _, _, w_u, b_u, _, w_l, b_l = inputs

        def func(variables):
            w_u, b_u, w_l, b_l, w_out_u, b_out_u, w_out_l, b_out_l = variables
            if len(w_u.shape) == 2:
                return w_out_u, b_out_u, w_out_l, b_out_l
            z_value = K.cast(0.0, w_u.dtype)
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

        lambda_ = Lambda(func, dtype=w_u.dtype)

        w_out_u_, b_out_u_, w_out_l_, b_out_l_ = lambda_([w_u, b_u, w_l, b_l] + back_bounds)

        if mode == F_FORWARD.name:
            return [inputs[0], w_out_u_, b_out_u_, w_out_l_, b_out_l_]
        else:
            u_out_c = upper_layer([x_tensor, w_out_u_, b_out_u_])
            l_out_c = lower_layer([x_tensor, w_out_l_, b_out_l_])

            return [inputs[0], u_out_c, w_out_u_, b_out_u_, l_out_c, w_out_l_, b_out_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")


def pre_process_inputs(input_layer_, mode, **kwargs):

    if mode == F_IBP.name:
        return input_layer_
    # convert input_layer_ according
    elif mode in {F_HYBRID.name, F_FORWARD.name}:
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

        if mode == F_FORWARD.name:
            x, w_u, b_u, w_l, b_l = input_layer_
            u_c_out, l_c_out = [upper_layer([x, w_u, b_u]), lower_layer([x, w_u, b_u])]

        else:
            x, u_c, w_u, b_u, l_c, w_l, b_l = input_layer_
            u_c_out, l_c_out = [
                Minimum()([u_c, upper_layer([x, w_u, b_u])]),
                Maximum()([l_c, lower_layer([x, w_u, b_u])]),
            ]

        if "flatten" in kwargs:
            op_flatten = kwargs["flatten"]
        else:
            op_flatten = Flatten()

        u_c_out_flatten = op_flatten(u_c_out)
        l_c_out_flatten = op_flatten(l_c_out)

        if "concatenate" in kwargs:
            op_concat = kwargs["concatenate"]
        else:
            op_concat = Concatenate(axis=1)

        x_ = op_concat([l_c_out_flatten[:, None], u_c_out_flatten[:, None]])
        z_value = K.cast(0.0, u_c_out.dtype)

        def func_create_weights(z):
            return tf.linalg.diag(z_value * z)

        init_weights = Lambda(func_create_weights, dtype=u_c_out.dtype)

        if mode == F_FORWARD.name:
            input_model = [x_, init_weights(u_c_out), u_c_out, init_weights(l_c_out), l_c_out]
        else:
            input_model = [x_, u_c_out, init_weights(u_c_out), u_c_out, l_c_out, init_weights(l_c_out), l_c_out]

        return input_model
    else:
        raise ValueError(f"Unknown mode {mode}")


def get_depth_dict(model):

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    nodes_list = []

    dico_depth = {}
    dico_nodes = {}

    def fill_dico(node, dico_depth=None):
        if dico_depth is None:
            dico_depth = {}

        parents = node.parent_nodes
        if len(parents):
            for parent in parents:
                if id(parent) not in dico_depth:
                    dico_depth = fill_dico(parent, dico_depth)

                depth = np.min([dico_depth[id(parent)] - 1 for parent in parents])
                if id(node) in dico_depth:
                    dico_depth[id(node)] = max(depth, dico_depth[id(node)])
                else:
                    dico_depth[id(node)] = depth
        else:
            dico_depth[id(node)] = max(depth_keys)

        return dico_depth

    for depth in depth_keys:
        # check for nodes that do not have parents and set depth to maximum depth value
        nodes = model._nodes_by_depth[depth]
        nodes_list += nodes
        for node in nodes:
            dico_depth = fill_dico(node, dico_depth)

    for node in nodes_list:
        depth = dico_depth[id(node)]
        if depth in dico_nodes:
            dico_nodes[depth].append(node)
        else:
            dico_nodes[depth] = [node]

    return dico_nodes


def get_inner_layers(model):

    count = 0
    for layer in model.layers:
        if isinstance(layer, Model):
            count += get_inner_layers(layer)
        else:
            count += 1
    return count


def convert_2_mode(mode_from, mode_to, convex_domain, dtype=K.floatx()):
    def get_2_mode_priv(inputs_):

        if mode_from == mode_to:
            return inputs_

        if mode_from in [F_FORWARD.name, F_HYBRID.name]:
            x_0 = inputs_[0]
        elif mode_from == F_IBP.name:
            u_c, l_c = inputs_
            if mode_to in [F_FORWARD.name, F_HYBRID.name]:
                x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
                z_value = K.cast(0.0, u_c.dtype)
                o_value = K.cast(1.0, u_c.dtype)
                w = tf.linalg.diag(z_value * l_c)
                b = z_value * l_c + o_value
                w_u = w
                b_u = b
                w_l = w
                b_l = b
            else:
                raise ValueError(f"Unknown mode {mode_to}")
        else:
            raise ValueError(f"Unknown mode {mode_from}")

        if mode_from == F_FORWARD.name:
            _, w_u, b_u, w_l, b_l = inputs_
            if mode_to in [F_IBP.name, F_HYBRID.name]:
                u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
                l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        elif mode_from == F_IBP.name:
            u_c, l_c = inputs_
        elif mode_from == F_HYBRID.name:
            _, u_c, w_u, b_u, l_c, w_l, b_l = inputs_
        else:
            raise ValueError(f"Unknown mode {mode_from}")

        if mode_to == F_IBP.name:
            return [u_c, l_c]
        elif mode_to == F_FORWARD.name:
            return [x_0, w_u, b_u, w_l, b_l]
        elif mode_to == F_HYBRID.name:
            return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]
        raise ValueError(f"Unknown mode {mode_to}")

    return Lambda(get_2_mode_priv, dtype=dtype)
