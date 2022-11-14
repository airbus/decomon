from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    InputLayer,
    Lambda,
    Layer,
    Maximum,
    Minimum,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.layers.decomon_layers import DecomonMaximum, DecomonMinimum
from decomon.layers.utils import linear_to_softmax
from decomon.layers.utils import softmax_to_linear as softmax_2_linear

from ..backward_layers.backward_layers import get_backward as get_backward_
from ..utils import (
    F_FORWARD,
    F_HYBRID,
    F_IBP,
    backward_maximum,
    backward_minimum,
    get_lower_layer,
    get_lower_layer_box,
    get_upper_layer,
    get_upper_layer_box,
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


def update_input(backward_bound, input_tensors, mode, output_shape, reshape=False, **kwargs):

    if "final_mode" in kwargs:
        final_mode = kwargs["final_mode"]
    else:
        final_mode = mode

    w_out_u, b_out_u, w_out_l, b_out_l = backward_bound
    if reshape:
        op_reshape_u = Reshape(list(output_shape)[1:])
        if mode in [F_HYBRID.name, F_FORWARD.name]:
            op_reshape_w = Reshape([-1] + list(output_shape)[1:])
    else:
        n_out = b_out_u.shape[-1]
        op_reshape_u = Reshape((n_out,))
        if mode in [F_HYBRID.name, F_FORWARD.name]:
            op_reshape_w = Reshape((-1, n_out))

    if mode == F_HYBRID.name:
        x, u_c, w_u, b_u, l_c, w_l, b_l = input_tensors
    if mode == F_FORWARD.name:
        x, w_u, b_u, w_l, b_l = input_tensors
    if mode == F_IBP.name:
        u_c, l_c = input_tensors

    if mode == F_IBP.name:

        # case 1: we do not change the mode
        if final_mode in [F_IBP.name, F_HYBRID.name]:

            if "flatten" in kwargs:
                op_flatten = kwargs["flatten"]
            else:
                op_flatten = Flatten()

            u_c_flat = op_flatten(u_c)
            l_c_flat = op_flatten(l_c)

            upper_layer_ibp = get_upper_layer_box()
            lower_layer_ibp = get_lower_layer_box()
            u_c_out = op_reshape_u(upper_layer_ibp([l_c_flat, u_c_flat, w_out_u, b_out_u]))
            l_c_out = op_reshape_u(lower_layer_ibp([l_c_flat, u_c_flat, w_out_l, b_out_l]))

        if final_mode in [F_HYBRID.name, F_FORWARD.name]:

            # create a fake x
            if "reshape" in kwargs:
                op_reshape = kwargs["reshape"]
            else:
                op_reshape = Reshape((1, -1))

            u_c_flat = op_reshape(u_c)
            l_c_flat = op_reshape(l_c)
            x = Concatenate(1)([l_c_flat, u_c_flat])

        if final_mode == F_IBP.name:
            return [u_c_out, l_c_out]
        if final_mode == F_FORWARD.name:
            return [x] + backward_bound
        if final_mode == F_HYBRID.name:
            return [x, u_c_out, w_out_u, b_out_u, l_c_out, w_out_l, b_out_l]

    # the mode is necessary with linear bounds

    def func(variables):

        w_u, b_u, w_l, b_l, w_out_u, b_out_u, w_out_l, b_out_l = variables
        if len(w_u.shape) == 2:
            # identity
            return w_out_u, b_out_u, w_out_l, b_out_l
        z_value = 0.0

        w_u_ = K.reshape(w_u, [-1, w_u.shape[1], np.prod(w_u.shape[2:])])  # (None, n_x, n_in)
        w_l_ = K.reshape(w_l, [-1, w_l.shape[1], np.prod(w_l.shape[2:])])  # (None, n_x, n_in)
        b_u_ = K.reshape(b_u, [-1, np.prod(b_u.shape[1:])])  # (None, n_in)
        b_l_ = K.reshape(b_l, [-1, np.prod(b_l.shape[1:])])  # (None, n_in)

        w_u_pos = K.maximum(w_out_u, z_value)  # (None, n_in, n_out)
        w_u_neg = K.minimum(w_out_u, z_value)

        w_l_pos = K.maximum(w_out_l, z_value)
        w_l_neg = K.minimum(w_out_l, z_value)

        #                sum( (None, 1, n_in, n_out)   *  (None, n_x, n_in, 1) -> (None, n_x, n_in, n_out), 2)- >(None, n_x, n_out)
        w_out_u_0 = K.sum(K.expand_dims(w_u_pos, 1) * K.expand_dims(w_u_, -1), 2) + K.sum(
            K.expand_dims(w_u_neg, 1) * K.expand_dims(w_l_, -1), 2
        )

        w_out_l_0 = K.sum(K.expand_dims(w_l_pos, 1) * K.expand_dims(w_l_, -1), 2) + K.sum(
            K.expand_dims(w_l_neg, 1) * K.expand_dims(w_u_, -1), 2
        )

        b_out_u_0 = K.sum(w_u_pos * K.expand_dims(b_u_, -1), 1) + K.sum(w_u_neg * K.expand_dims(b_l_, -1), 1) + b_out_u
        b_out_l_0 = K.sum(w_l_pos * K.expand_dims(b_l_, -1), 1) + K.sum(w_l_neg * K.expand_dims(b_u_, -1), 1) + b_out_l

        return w_out_u_0, b_out_u_0, w_out_l_0, b_out_l_0

    lambda_ = Lambda(lambda var: func(var))

    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = lambda_([w_u, b_u, w_l, b_l] + backward_bound)
    w_out_u_ = op_reshape_w(w_out_u_)
    w_out_l_ = op_reshape_w(w_out_l_)
    b_out_u_ = op_reshape_u(b_out_u_)
    b_out_l_ = op_reshape_u(b_out_l_)

    if final_mode in [F_HYBRID.name, F_IBP.name]:
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
        u_c_out = upper_layer([x, w_out_u_, b_out_u_])
        l_c_out = lower_layer([x, w_out_l_, b_out_l_])

    if final_mode == F_IBP.name:
        return [u_c_out, l_c_out]
    if final_mode == F_FORWARD.name:
        return [x, w_out_u_, b_out_u_, w_out_l_, b_out_l_]
    if final_mode == F_HYBRID.name:
        return [x, u_c_out, w_out_u_, b_out_u_, l_c_out, w_out_l_, b_out_l_]


def pre_process_inputs(input_layer_, mode, **kwargs):

    if mode == F_IBP.name:
        input_model = input_layer_
    # convert input_layer_ according
    if mode in [F_HYBRID.name, F_FORWARD.name]:
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

        u_c_out, l_c_out = [upper_layer(x, w_u, b_u), lower_layer(x, w_u, b_u)]

    if mode == F_HYBRID.name:
        x, u_c, w_u, b_u, l_c, w_l, b_l = input_layer_
        u_c_out, l_c_out = [Minimum()([u_c, upper_layer([x, w_u, b_u])]), Maximum()([l_c, lower_layer([x, w_u, b_u])])]

    if "flatten" in kwargs:
        op_flatten = kwargs["flatten"]
    else:
        op_flatten = Flatten()
    shape = u_c_out.shape[1:]
    u_c_out_flatten = op_flatten(u_c_out)
    l_c_out_flatten = op_flatten(l_c_out)

    if "concatenate" in kwargs:
        op_concat = kwargs["concatenate"]
    else:
        op_concat = Concatenate(axis=1)

    x_ = op_concat([l_c_out_flatten[:, None], u_c_out_flatten[:, None]])
    z_value = K.cast(0.0, K.floatx())

    def func_create_weights(z):
        return tf.linalg.diag(z_value * z)

    init_weights = Lambda(func_create_weights)

    if mode == F_FORWARD.name:
        input_model = [x_, init_weights(u_c_out), u_c_out, init_weights(l_c_out), l_c_out]
    if mode == F_HYBRID.name:
        input_model = [x_, u_c_out, init_weights(u_c_out), u_c_out, l_c_out, init_weights(l_c_out), l_c_out]

    return input_model


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

    lambda_f = Lambda(lambda x: func(x))
    output = lambda_f(max_bounds + min_bounds)
    return output


def get_backward_layer_input(
    layer_, id_node, forward_map, mode, back_bounds, finetune, convex_domain, input_dim, rec=1
):

    # assume back_bounds is either an empty dict

    layer_back = get_backward_(
        layer_,
        previous=(len(back_bounds) > 0),
        mode=mode,
        finetune=finetune,
        convex_domain=convex_domain,
        input_dim=input_dim,
        rec=rec,
    )
    input_layer_ = forward_map["{}_{}".format(layer_.name, id_node)]
    back_bounds_ = layer_back(input_layer_ + back_bounds)
    if isinstance(back_bounds_, tuple):
        back_bounds_ = list(back_bounds_)

    return {"{}_{}".format(layer_.name, id_node): back_bounds}


def get_output_model(node, layer_map, forward_map, mode, input_dim, rec=1, **kwargs):

    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)
    if "{}_{}".format(layer_.name, id_node) in forward_map:
        if isinstance(layer_, Model):
            return forward_map["{}_{}".format(layer_.name, id_node)][0], False
        else:
            return forward_map["{}_{}".format(layer_.name, id_node)], False
    else:
        if len(to_list(layer_.output_shape)) > 1:
            raise NotImplementedError("Decomon cannot handle nodes with a list of output tensors")
        backward_map = get_backward_layer(node, layer_map, forward_map, mode, [], input_dim, rec=rec, **kwargs)
        outputs = []
        for key in backward_map:
            tmp = update_input(backward_map[key], forward_map[key], mode, layer_.output_shape, reshape=True, **kwargs)
            outputs += tmp

        if "convex_domain" in kwargs:
            convex_domain = kwargs["convex_domain"]
        else:
            convex_domain = {}

        output_max = DecomonMinimum(mode=mode, convex_domain=convex_domain)(outputs)
        output_min = DecomonMaximum(mode=mode, convex_domain=convex_domain)(outputs)

        if mode == F_IBP.name:
            return [output_max[0], output_min[-1]], True
        if mode == F_FORWARD.name:
            return output_max[:3] + output_min[-2:], True
        if mode == F_HYBRID.name:
            return output_max[:4] + output_min[-3:], True


def get_output_layer(node, layer_map, forward_map, mode, input_dim, rec=1, **kwargs):

    layer_ = node.outbound_layer
    id_node = get_node_by_id(node)
    if "{}_{}".format(layer_.name, id_node) in forward_map:
        if isinstance(layer_, Model):
            return forward_map["{}_{}".format(layer_.name, id_node)][0]
        else:
            return forward_map["{}_{}".format(layer_.name, id_node)]
    else:
        if len(to_list(layer_.get_output_shape_at(0))) > 1:
            raise NotImplementedError("Decomon cannot handle nodes with a list of output tensors")
        backward_map = get_backward_layer(
            node, layer_map, forward_map, mode, [], input_dim, rec=rec, **kwargs
        )  # recursiv approach
        outputs = []
        if "convex_domain" in kwargs:
            convex_domain = kwargs["convex_domain"]
        else:
            convex_domain = {}
        kwargs["convex_domain"] = convex_domain
        for key in backward_map:
            tmp = update_input(
                backward_map[key], forward_map[key], mode, layer_.get_output_shape_at(0), reshape=True, **kwargs
            )
            outputs += tmp

        if len(backward_map.keys()) > 1:
            output_max = DecomonMinimum(mode=mode, convex_domain=convex_domain)(outputs)
            output_min = DecomonMaximum(mode=mode, convex_domain=convex_domain)(outputs)
        else:
            output_max = outputs
            output_min = outputs

        if mode == F_IBP.name:
            forward_map["{}_{}".format(layer_.name, id_node)] = [output_max[0], output_min[-1]]
            return [output_max[0], output_min[-1]]
        if mode == F_FORWARD.name:
            forward_map["{}_{}".format(layer_.name, id_node)] = output_max[:3] + output_min[-2:]
            return output_max[:3] + output_min[-2:]
        if mode == F_HYBRID.name:
            forward_map["{}_{}".format(layer_.name, id_node)] = output_max[:4] + output_min[-3:]
            return output_max[:4] + output_min[-3:]


def get_fake_input(layer_, mode):
    # get input_shape
    n_in = layer_.get_input_shape_at(0)[1:]
    vec = K.zeros([1] + to_list(n_in))
    f = lambda x: vec
    return [Lambda(f)()]  # enough ?


def get_backward_layer(node, layer_map, forward_map, mode, back_bounds, input_dim, rec=1, **kwargs):

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
        return get_backward_layer_input(
            layer_, id_node, forward_map, mode, back_bounds, finetune, convex_domain, input_dim, rec=rec
        )

    if "{}_{}".format(layer_.name, id_node) not in layer_map or isinstance(layer_, Model):
        layer_list = [layer_]
    else:
        layer_list = to_list(layer_map["{}_{}".format(layer_.name, id_node)])
        layer_list = layer_list[::-1]

    for i in range(len(layer_list) - 1):
        # if layer_list has more than one element, then there has been a forward pass
        # hence forward_map contains the related input
        if "{}_{}".format(layer_list[i + 1].name, id_node) not in forward_map:
            # split name
            new_name = "{}{}".format(layer_list[i + 1].name.split("_monotonic")[0], "_monotonic")
            inputs_i = forward_map["{}_{}".format(new_name, id_node)]
        else:
            inputs_i = forward_map["{}_{}".format(layer_list[i + 1].name, id_node)]

        back_layer_i = get_backward_(
            layer_list[i],
            previous=(len(back_bounds) > 0),
            mode=mode,
            finetune=finetune,
            convex_domain=convex_domain,
            input_dim=input_dim,
            rec=rec,
        )
        back_bounds_ = back_layer_i(inputs_i + back_bounds)
        if isinstance(back_bounds_, tuple):
            back_bounds_ = list(back_bounds_)
        back_bounds = back_bounds_

    layer_ = layer_list[-1]

    input_nodes = to_list(node.parent_nodes)
    # get input nodes

    if isinstance(layer_, Model):
        if len(to_list(layer_.output_shape)) > 1:
            raise NotImplementedError("Decomon cannot handle nodes with a list of output tensors")

        input_model_ = []
        for node_i in input_nodes:
            tmp, crown_flag = get_output_model(node_i, layer_map, forward_map, mode, input_dim, rec=1, **kwargs)

            # if crown_flag and isinstance(layer_, Model):
            #    import pdb;
            #    pdb.set_trace()
            #    tmp = pre_process_inputs(tmp, mode)
            input_model_.append(tmp)

        finetune_ = False
        convex_domain_ = {}

        if "finetune" in kwargs:
            finetune_ = kwargs["finetune"]
        if "convex_domain" in kwargs:
            convex_domain_ = kwargs["convex_domain"]
        if "{}_{}".format(layer_.name, id_node) in layer_map:
            l_map = layer_map["{}_{}".format(layer_.name, id_node)]
        else:
            l_map = {}

        if "{}_{}".format(layer_.name, id_node) in forward_map:
            f_map = forward_map["{}_{}".format(layer_.name, id_node)][-1]
        else:
            f_map = {}

        kwargs_ = dict(
            [
                (key, kwargs[key])
                for key in kwargs
                if key
                not in ["forward_map", "layer_map", "convex_domain", "IBP", "forward", "finetune", "fuse_with_input"]
            ]
        )

        back_bounds_ = get_backward_model(
            layer_,
            input_tensors=input_model_,
            back_bounds=back_bounds,
            convex_domain=convex_domain_,
            IBP=get_IBP(mode),
            forward=get_FORWARD(mode),
            finetune=finetune_,
            fuse_with_input=False,
            layer_map=l_map,
            forward_map=f_map,
            **kwargs_,
            rec=rec,
        )[1]
        # if len(layer_.input_shape)>1:
        #    back_bounds_ = back_bounds_[0]

    else:
        input_layer_ = []

        """
        if not isinstance(layer_, Conv2D) or layer_.activation_name!='linear' or not len(back_bounds): # Linear layer do not need recursive calls
            for node_i in input_nodes:
                tmp = get_output_layer(node_i, layer_map, forward_map, mode, input_dim, **kwargs)

                input_layer_ += tmp
        else:
            # generate fake inputs
            input_layer_ = []
        """
        back_layer = get_backward_(
            layer_,
            previous=(len(back_bounds) > 0),
            mode=mode,
            finetune=finetune,
            convex_domain=convex_domain,
            input_dim=input_dim,
            rec=rec,
        )

        if (
            not isinstance(layer_, Dense)
            or not isinstance(layer_, Conv2D)
            or layer_.get_config()["activation"] != "linear"
            or not len(back_bounds)
        ):  # Linear layer do not need recursive calls
            for node_i in input_nodes:
                tmp = get_output_layer(node_i, layer_map, forward_map, mode, input_dim, rec=rec, **kwargs)

                input_layer_ += tmp
        else:
            # generate fake inputs
            input_layer_ = kwargs["fake_input"]
            """
            for node_i in input_nodes:
                tmp = get_output_layer(node_i, layer_map, forward_map, mode, input_dim, **kwargs)

                input_layer_ += tmp
            """

        back_bounds_ = back_layer(input_layer_ + back_bounds)
    if isinstance(back_bounds_, tuple):
        back_bounds_ = list(back_bounds_)
    back_bounds = back_bounds_

    if len(input_nodes) == 1:
        return get_backward_layer(
            input_nodes[0], layer_map, forward_map, mode, back_bounds, input_dim=input_dim, rec=rec + 1, **kwargs
        )
    else:
        dico_backward = {}
        for node_i, bound_i in zip(input_nodes, back_bounds):
            backward_map_i = get_backward_layer(
                node_i, layer_map, forward_map, mode, bound_i, input_dim=input_dim, rec=rec + 1, **kwargs
            )

            for key_i in backward_map_i:
                if key_i not in dico_backward:
                    dico_backward[key_i] = [backward_map_i[key_i]]
                else:
                    dico_backward[key_i].append(backward_map_i[key_i])

        # fuse several elements if needed
        for key in dico_backward:
            dico_backward[key] = fuse_backward_bounds(dico_backward[key], forward_map[key], mode, **kwargs)
        return dico_backward


def get_backward_model(
    model,
    input_tensors=None,
    back_bounds=[],
    input_dim=-1,
    convex_domain={},
    IBP=True,
    forward=True,
    finetune=False,
    layer_map={},
    forward_map={},
    fuse_with_input=True,
    softmax_to_linear=True,
    rec=1,
    **kwargs,
):
    if not isinstance(model, Model):
        raise ValueError()

    has_softmax = False
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

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
            if "{}_{}".format(layer_i, id_node_i) not in forward_map:
                if input_dim == -1:
                    input_dim_i = np.prod(layer_i.input_shape[1:])
                else:
                    input_dim_i = input_dim
                if "{}_{}".format(layer_i.name, id_node_i) not in forward_map:
                    forward_map["{}_{}".format(layer_i.name, id_node_i)] = check_input_tensors_sequential(
                        model, None, input_dim_i, input_dim_init, IBP, forward, False, convex_domain
                    )
    else:
        if not isinstance(input_tensors[0], list):
            input_tensors = [input_tensors]
        for i, node_i in enumerate(nodes_input):
            layer_i = node_i.outbound_layer
            id_node_i = get_node_by_id(node_i)
            if "{}_{}".format(layer_i.name, id_node_i) not in forward_map:
                forward_map["{}_{}".format(layer_i.name, id_node_i)] = input_tensors[i]

            input_dim = input_tensors[i][0].shape[-1]

    mode = get_mode(IBP=IBP, forward=forward)
    if "final_ibp" in kwargs or "final_forward" in kwargs:
        final_IBP = False
        final_forward = False

        if "final_ibp" in kwargs:
            final_IBP = kwargs["final_ibp"]
        if "final_forward" in kwargs:
            final_forward = kwargs["final_forward"]
        if final_IBP or final_forward:
            final_mode = get_mode(final_IBP, final_forward)
        else:
            final_mode = mode
    else:
        final_mode = None

    if "mode_output" not in kwargs:
        mode_output = mode
    else:
        mode_output = kwargs["mode_output"]

    if not isinstance(back_bounds, list):
        import pdb

        pdb.set_trace()

    if len(back_bounds) and not isinstance(back_bounds[0], list):
        back_bounds = [back_bounds]

    if not len(back_bounds) in [0, nb_output]:
        import pdb

        pdb.set_trace()

    if len(back_bounds) == 0:
        back_bounds = [[]] * nb_output

    nodes_output = model._nodes_by_depth[0]
    output = []
    for node_i, bounds_i in zip(nodes_output, back_bounds):
        if len(bounds_i) == 1:
            w_u = bounds_i[0]
            w_l = w_u
            b_u = 0 * w_u[:, 0]
            b_l = b_u
            bounds_i = [w_u, b_u, w_l, b_l]

        back_bound_i_dict = get_backward_layer(
            node_i,
            layer_map,
            forward_map,
            mode=mode,
            back_bounds=to_list(bounds_i),
            input_dim=input_dim,
            convex_domain=convex_domain,
            finetune=finetune,
            rec=rec,
            **kwargs,
        )
        back_bound_i = []

        for node_j in nodes_input:
            layer_j = node_j.outbound_layer
            layer_name_j = layer_j.name
            id_j = get_node_by_id(node_j)

            if "{}_{}".format(layer_name_j, id_j) not in back_bound_i_dict:
                raise NotImplementedError()
            else:
                output_i = back_bound_i_dict["{}_{}".format(layer_name_j, id_j)]
                if not fuse_with_input:
                    back_bound_i.append(output_i)
                else:
                    inputs_tensors_i = forward_map["{}_{}".format(layer_name_j, id_j)]
                    kwargs["convex_domain"] = convex_domain
                    if final_mode:
                        # do not return the same mode as the one used in the internal layers
                        output_i = update_input(
                            output_i,
                            inputs_tensors_i,
                            mode,
                            node_i.outbound_layer.output_shape,
                            final_mode=final_mode,
                            **kwargs,
                        )
                    else:
                        output_i = update_input(
                            output_i, inputs_tensors_i, mode, node_i.outbound_layer.output_shape, **kwargs
                        )

                    """
                    if "{}_{}".format(node_i.outbound_layer.name, get_node_by_id(node_i)) in forward_map:
                        output_tmp_i = forward_map["{}_{}".format(node_i.outbound_layer.name, get_node_by_id(node_i))]
                        max_bounds = DecomonMinimum(mode=mode)(output_tmp_i+output_i)
                        min_bounds = DecomonMaximum(mode=mode)(output_tmp_i+output_i)

                        def func(inputs_):
                            n = int(len(inputs_) / 2)
                            inputs_0 = inputs_[:n]
                            inputs_1 = inputs_[n:]

                            if mode == F_IBP.name:
                                u_c = inputs_0[0]
                                l_c = inputs_1[-1]
                                return [u_c, l_c]
                            if mode == F_FORWARD.name:
                                x, w_u, b_u = inputs_0[:3]
                                w_l, b_l = inputs_1[-2:]
                                return [x, w_u, b_u, w_l, b_l]
                            if mode == F_HYBRID.name:
                                x, u_c, w_u, b_u = inputs_0[:4]
                                l_c, w_l, b_l = inputs_1[-3:]
                                return [x, u_c, w_u, b_u, l_c, w_l, b_l]

                        lambda_f = Lambda(lambda x: func(x))
                        output_i = lambda_f(max_bounds + min_bounds)
                    """
                    forward_map["{}_{}".format(node_i.outbound_layer.name, get_node_by_id(node_i))] = output_i
                    # print("{}_{}".format(node_i.outbound_layer.name, get_node_by_id(node_i)))
                    back_bound_i.append(output_i)

        output += back_bound_i

    if has_softmax:
        linear_to_softmax(model)

    # here update layer_map annd forward_map
    if nb_output == 1 and nb_input == 1:
        output = output[0]

    inputs_ = []
    for node_j in nodes_input:
        layer_j = node_j.outbound_layer
        layer_name_j = layer_j.name
        id_j = get_node_by_id(node_j)
        inputs_ += forward_map["{}_{}".format(layer_name_j, id_j)]

    for elem in back_bounds:
        inputs_ += elem

    return inputs_, output, layer_map, forward_map


# Aliasing
convert_backward_model = get_backward_model
