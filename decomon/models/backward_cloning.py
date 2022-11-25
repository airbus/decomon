from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add, Average
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Dense, Dropout, Lambda, Permute, Reshape
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.layers.utils import get_lower, get_upper
from decomon.layers.utils import softmax_to_linear as softmax_2_linear

from ..backward_layers.backward_layers import get_backward as get_backward_
from ..backward_layers.backward_merge import BackwardMerge
from ..backward_layers.utils import merge_with_previous
from ..utils import F_FORWARD, F_HYBRID, F_IBP, get_lower, get_upper
from .utils import (
    check_input_tensors_sequential,
    convert_2_mode,
    get_depth_dict,
    get_mode,
)


def is_purely_linear(layer):
    return False


def get_input(node, input_tensors, forward_map, output_map):

    parents = node.parent_nodes

    if len(parents) == 0:
        return input_tensors

    layer = node.outbound_layer

    if is_purely_linear(layer):
        return []

    output = []
    for parent in parents:

        if id(parent) in output_map:
            output += output_map[id(parent)]
        else:
            raise NotImplementedError()


"""
def get_backward_from_forward(model,
                              input_tensors=None,
                              back_bounds=[],
                              output_map,
                              forward_map,
                              backward_map,
                              finetune=False,
                              joint=True,
                              layer_fn=None,
                              ):

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

        # create input tensors

        # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    output = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:

            layer = node.outbound_layer
            # get input
            parents = node.parent_nodes
            if len(parents):
                output = []
                for parent in parents:
                    output += output_map[id(parent)]

            # retrieve layers
            if id(node) in forward_map.keys
                if id(node) in output_map.keys() and joint:
                    continue




def get_backward_model(
    model,
    input_tensors=None,
    back_bounds=[],
    input_dim=-1,
    convex_domain=None,
    IBP=True,
    forward=True,
    finetune=False,
    layer_fn = None,
    backward_map=None,
    forward_map=None,
    output_map=None,
    finetune=False,
    joint=True,
    rec=1,
    input_tensors=None
    **kwargs,
):

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")

        # create input tensors

        # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)


    output = input_tensors
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:

            layer = node.outbound_layer
            # get input
            parents = node.parent_nodes
            if len(parents):
                output=[]
                for parent in parents:
                    output+=output_map[id(parent)]

            # retrieve layers
            if id(node) in forward_map.keys
            if id(node) in output_map.keys() and joint:
                continue
"""

""""
def get_fuse(mode):

    def get_fuse_priv(inputs_):

        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bounds

        if mode == F_FORWARD.name:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs
        elif mode == F_HYBRID.name:
            x_0, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l = inputs
        else:
            u_c, l_c = inputs


        if mode in [F_HYBRID.name, F_IBP.name]:
            u_c_ = get_upper_box(l_c, u_c, w_out_u, b_out_u)
            l_c_ = get_lower_box(l_c, u_c, w_out_l, b_out_l)

        if mode in [F_HYBRID.name, F_FORWARD.name]:

            w_u, b_u, w_l, b_l = merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l]+backward_bounds)

        if mode == F_IBP.name:
            return [u_c_, l_c_]
        if mode == F_FORWARD.name:
            return [x_0, w_u, b_u, w_l, b_l]

        if mode == F_HYBRID.name:
            return [x_0, u_c_, w_u, b_u, l_c_, w_l, b_l]

    return LambdaLayer(get_fuse_priv())


def get_no_fuse(mode, convex_domain):

    def get_fuse_priv(inputs_):

        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bounds

        if mode == F_FORWARD.name:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs
            u_c_ = get_upper(x_0, w_f_u, b_f_u, convex_domain)
            l_c_ = get_lower(x_0, w_f_l, b_f_l, convex_domain)
        elif mode == F_HYBRID.name:
            _, u_c, _, _, l_c, _, _ = inputs
        else:
            u_c, l_c = inputs

        x_ = K.concatenate([K.expand_dims(l_c_, 1), K.expand_dims(u_c_, 1)], 1)




        if mode in [F_HYBRID.name, F_IBP.name]:
            u_c_ = get_upper_box(l_c, u_c, w_out_u, b_out_u)
            l_c_ = get_lower_box(l_c, u_c, w_out_l, b_out_l)

        if mode in [F_HYBRID.name, F_FORWARD.name]:

            w_u, b_u, w_l, b_l = merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l]+backward_bounds)

        if mode == F_IBP.name:
            return [u_c_, l_c_]
        if mode == F_FORWARD.name:
            return [x_0, w_u, b_u, w_l, b_l]

        if mode == F_HYBRID.name:
            return [x_0, u_c_, w_u, b_u, l_c_, w_l, b_l]

    return LambdaLayer(get_fuse_priv())
"""


def get_fuse(mode, dtype=K.floatx()):
    def get_fuse_priv(inputs_):

        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]

        if mode == F_FORWARD.name:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs
        elif mode == F_HYBRID.name:
            x_0, u_c, w_f_u, b_f_u, l_c, w_f_l, b_f_l = inputs
        else:
            return backward_bounds

        return merge_with_previous([w_f_u, b_f_u, w_f_l, b_f_l] + backward_bounds)

    return Lambda(get_fuse_priv, dtype=dtype)


def convert_backward_2_mode(mode, convex_domain, dtype=K.floatx()):
    def get_2_mode_priv(inputs_):

        inputs = inputs_[:-4]
        backward_bounds = inputs_[-4:]
        w_out_u, b_out_u, w_out_l, b_out_l = backward_bounds

        if mode in [F_FORWARD.name, F_HYBRID.name]:
            x_0 = inputs[0]
        else:
            u_c, l_c = inputs
            x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)

        if mode == F_FORWARD.name:
            return [x_0] + backward_bounds

        if mode == F_IBP.name:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain={})
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain={})
            return [u_c_, l_c_]

        if mode == F_HYBRID.name:
            u_c_ = get_upper(x_0, w_out_u, b_out_u, convex_domain=convex_domain)
            l_c_ = get_lower(x_0, w_out_l, b_out_l, convex_domain=convex_domain)
            return [x_0, u_c_, w_out_u, b_out_u, l_c_, w_out_l, b_out_l]

    return Lambda(get_2_mode_priv, dtype=dtype)

    # return convert_2_mode(mode_from=F_FORWARD.name, mode_to=mode, convex_domain=convex_domain)


def get_disconnected_input(mode, convex_domain, dtype=K.floatx()):
    def disco_priv(inputs_):

        if mode == F_IBP.name:
            return inputs_
        elif mode in {F_FORWARD.name, F_HYBRID.name}:
            if mode == F_FORWARD.name:
                x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs_
                u_c = get_upper(x_0, w_f_u, b_f_u, convex_domain=convex_domain)
                l_c = get_lower(x_0, w_f_l, b_f_l, convex_domain=convex_domain)
            else:
                _, u_c, _, _, l_c, _, _ = inputs_

            x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            w_u_ = tf.linalg.diag(K.cast(0.0, x_0.dtype) * u_c + K.cast(1.0, x_0.dtype))
            b_u_ = K.cast(0.0, x_0.dtype) * u_c
            # w_u_ = tf.linalg.diag(K.cast(0., x_0.dtype)*u_c)

            if mode == F_FORWARD.name:
                return [x_0, w_u_, b_u_, w_u_, b_u_]
            else:
                return [x_0, u_c, w_u_, b_u_, l_c, w_u_, b_u_]
        else:
            raise ValueError(f"Unknown mode {mode}")

    return Lambda(disco_priv, dtype=dtype)


def retrieve_layer(node, previous, layer_fn, backward_map, joint=True):
    def retrieve_layer_priv(node, previous, pattern, layer_fn, backward_map, joint):
        if id(node) in backward_map.keys() and pattern in backward_map[id(node)].keys():
            backward_layer = backward_map[id(node)][pattern]
        else:
            backward_layer = layer_fn(node.outbound_layer)
            backward_layer.previous = bool(previous)
            if joint:
                # no need to create another object at the next call
                if id(node) not in backward_map.keys():
                    backward_map[id(node)] = {pattern: backward_layer}
                else:
                    backward_map[id(node)][pattern] = backward_layer
        return backward_layer, backward_map

    if previous:
        return retrieve_layer_priv(node, previous, "previous", layer_fn, backward_map, joint)
    else:
        return retrieve_layer_priv(node, previous, "no_previous", layer_fn, backward_map, joint)


def crown_(
    node,
    IBP,
    forward,
    convex_domain,
    input_map,
    layer_fn,
    backward_bounds,
    backward_map=None,
    joint=True,
    fuse=True,
    output_map=None,
    merge_layers=None,
    **kwargs,
):
    """

    :param node:
    :param IBP:
    :param forward:
    :param input_map:
    :param layer_fn:
    :param backward_bounds:
    :param backward_map:
    :param joint:
    :param fuse:
    :return: list of 4 tensors affine upper and lower bounds
    """
    if backward_map is None:
        backward_map = {}

    if output_map is None:
        output_map = {}

    inputs = input_map[id(node)]

    if convex_domain is None:
        convex_domain = {}

    if merge_layers is None:
        merge_layers = merge_with_previous

    if isinstance(node.outbound_layer, Model):

        inputs_ = get_disconnected_input(get_mode(IBP, forward), convex_domain, dtype=inputs[0].dtype)(inputs)
        kwargs.update({"debug": True})
        _, backward_bounds_, backward_map_, _ = crown_model(
            node.outbound_layer,
            input_tensors=inputs_,
            backward_bounds=backward_bounds,
            IBP=IBP,
            forward=forward,
            convex_domain=None,
            finetune=False,
            joint=joint,
            fuse=False,
            **kwargs,
        )

    else:
        backward_layer, backward_map_ = retrieve_layer(node, len(backward_bounds), layer_fn, backward_map, joint)
        backward_map.update(backward_map_)
        # to retrieve
        backward_bounds_ = backward_layer.call_no_previous(inputs)
        # update back_bounds
        try:
            # backward_layer.previous=False
            # backward_bounds_ = backward_layer.call_no_previous(inputs)

            if id(node) not in output_map.keys():
                backward_bounds_ = backward_layer.call_no_previous(inputs)
                output_map[id(node)] = backward_bounds_
            else:
                backward_bounds_ = output_map[id(node)]

            # backward_bounds_ = backward_layer.call_no_previous(inputs)

            if not isinstance(backward_bounds_, list):
                backward_bounds_ = [e for e in backward_bounds_]
            # import pdb; pdb.set_trace()
            if len(backward_bounds):
                backward_tmp = merge_layers(backward_bounds_ + backward_bounds)
                backward_bounds_ = backward_tmp

            # backward_bounds_ = \
            #    backward_layer(inputs+backward_bounds)
            # debugging

            # backward_layer.previous = False
            # tmp = backward_layer.call_no_previous(inputs)
        except ValueError:
            import pdb

            pdb.set_trace()
    if not isinstance(backward_bounds_, list):
        backward_bounds = [e for e in backward_bounds_]
    else:
        backward_bounds = backward_bounds_

    parents = node.parent_nodes

    if len(parents):

        """
        if isinstance(node.layer, _Merge):
            crown_bounds=[]
            for b_bounds, parent in zip(backward_bounds, parents):
                tmp, backward_map_ = crown_(parent, IBP, forward, input_map, layer_fn, b_bounds, backward_map, joint, fuse)
                backward_map.update(backward_map_)
                crown_bounds.append(tmp)

            return crown_bounds[0], backward_map #TODO optimize with minimum !!!
        else:
            if len(parents)>1:
                raise NotImplementedError()
            crown_bound, backward_map_ = crown_(parents[0], IBP, forward, input_map, layer_fn, backward_bounds, backward_map, joint, fuse)
            backward_map.update(backward_map_)
            return crown_bound, backward_map
        """
        if len(parents) > 1:
            raise NotImplementedError()
        else:
            crown_bound, backward_map_, output_map_ = crown_(
                parents[0],
                IBP,
                forward,
                convex_domain,
                input_map,
                layer_fn,
                backward_bounds,
                backward_map,
                joint,
                fuse,
                output_map=output_map,
                merge_layers=merge_layers,
            )
            backward_map.update(backward_map_)
            output_map.update(output_map_)
        return crown_bound, backward_map, output_map
    else:
        # do something
        if fuse:
            fuse_layer = get_fuse(get_mode(IBP=IBP, forward=forward), dtype=inputs[0].dtype)
            result = fuse_layer(inputs + backward_bounds)

            return result, backward_map, output_map

        else:
            return backward_bounds, backward_map, output_map


def get_input_nodes(
    model,
    dico_nodes,
    IBP,
    forward,
    input_tensors,
    output_map,
    layer_fn,
    joint,
    convex_domain=None,
    merge_layers=None,
    **kwargs,
):

    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)
    input_map = {}
    backward_map = {}
    if convex_domain is None:
        convex_domain = {}
    crown_map = {}
    set_mode_layer = None
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            parents = node.parent_nodes
            if not len(parents):
                # if 'debug' in kwargs.keys():
                #    import pdb; pdb.set_trace()
                input_map[id(node)] = input_tensors
            else:
                if is_purely_linear(node.outbound_layer):
                    input_map[id(node)] = []
                else:
                    output = []
                    for parent in parents:
                        # do something
                        if id(parent) in output_map.keys():
                            output += output_map[id(parent)]
                        else:
                            # if 'debug' in kwargs.keys():
                            #    import pdb;
                            #    pdb.set_trace()
                            if merge_layers is None:
                                merge_layers = merge_with_previous
                            output_crown, backward_map_, crown_map_ = crown_(
                                parent,
                                IBP=IBP,
                                forward=forward,
                                input_map=input_map,
                                layer_fn=layer_fn,
                                backward_bounds=[],
                                backward_map=backward_map,
                                joint=joint,
                                fuse=True,
                                convex_domain=convex_domain,
                                output_map=crown_map,
                                merge_layers=merge_layers,
                            )
                            # if 'debug' in kwargs.keys():
                            #    pdb.set_trace()
                            backward_map.update(backward_map_)
                            crown_map.update(crown_map_)

                            # convert output_crown in the right mode
                            try:
                                if set_mode_layer is None:
                                    set_mode_layer = convert_backward_2_mode(
                                        get_mode(IBP, forward), convex_domain, dtype=input_tensors[0].dtype
                                    )
                                output_crown_ = set_mode_layer(input_tensors + output_crown)
                                output += to_list(output_crown_)
                                # crown_map[id(parent)]=output_crown_

                            except TypeError:
                                import pdb

                                pdb.set_trace()
                    input_map[id(node)] = output
    return input_map, backward_map, crown_map


def crown_model(
    model,
    input_tensors=None,
    back_bounds=None,
    input_dim=-1,
    IBP=True,
    forward=True,
    convex_domain=None,
    finetune=False,
    forward_map=None,
    softmax_to_linear=True,
    joint=True,
    layer_fn=get_backward_,
    fuse=True,
    **kwargs,
):
    if back_bounds is None:
        back_bounds = []
    if forward_map is None:
        forward_map = {}
    if not isinstance(model, Model):
        raise ValueError()
    # import time
    # zero_time = time.process_time()
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

    if input_dim == -1:
        if isinstance(model.input_shape, list):
            input_dim = np.prod(model.input_shape[0][1:])
        else:
            input_dim = np.prod(model.input_shape[1:])
    if input_tensors is None:
        # check that the model has one input else
        input_tensors = []
        for i in range(len(model._input_layers)):

            tmp = check_input_tensors_sequential(model, None, input_dim, input_dim, IBP, forward, False, convex_domain)
            input_tensors += tmp

    # layer_fn
    ##########
    has_iter = False
    if layer_fn is not None and len(layer_fn.__code__.co_varnames) == 1 and "layer" in layer_fn.__code__.co_varnames:
        has_iter = True

    if not has_iter:
        # layer, slope = V_slope.name, previous = True, mode = F_HYBRID.name, convex_domain = {}, finetune = False, rec = 1, ** kwargs
        def func(layer):
            return get_backward_(layer, mode=get_mode(IBP, forward))

        layer_fn = func

    if not callable(layer_fn):
        raise ValueError("Expected `layer_fn` argument to be a callable.")
    ###############

    if len(back_bounds) and len(to_list(model.output)) > 1:
        raise NotImplementedError()

    # sort nodes from input to output
    dico_nodes = get_depth_dict(model)
    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)

    # generate input_map
    if not finetune:
        joint = True

    input_map, backward_map, crown_map = get_input_nodes(
        model,
        dico_nodes,
        IBP,
        forward,
        input_tensors,
        forward_map,
        layer_fn,
        joint,
        convex_domain=convex_domain,
        **kwargs,
    )
    # time_1 = time.process_time()
    # print('step1', time_1-zero_time)
    # retrieve output nodes
    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.layer.name for tensor in to_list(model.output)]
    for output_name in output_names:
        for node in output_nodes:
            if node.outbound_layer.name == output_name:
                # compute with crown
                output_crown, _, _ = crown_(
                    node,
                    IBP=IBP,
                    forward=forward,
                    input_map=input_map,
                    layer_fn=layer_fn,
                    backward_bounds=back_bounds,
                    backward_map=backward_map,
                    joint=joint,
                    fuse=fuse,
                    convex_domain=convex_domain,
                    output_map=crown_map,
                )
                # time_2 = time.process_time()
                # print('step2', time_2-time_1)
                if fuse:
                    # import pdb; pdb.set_trace()
                    output += to_list(
                        convert_backward_2_mode(get_mode(IBP, forward), convex_domain, dtype=input_tensors[0].dtype)(
                            input_tensors + output_crown
                        )
                    )
                else:
                    output = output_crown

    return input_tensors, output, backward_map, None


def get_backward_model(
    model,
    input_tensors=None,
    back_bounds=None,
    input_dim=-1,
    IBP=True,
    forward=True,
    convex_domain=None,
    finetune=False,
    forward_map=None,
    softmax_to_linear=True,
    joint=True,
    layer_fn=get_backward_,
    final_ibp=True,
    final_forward=False,
    **kwargs,
):
    if back_bounds is None:
        back_bounds = []
    if forward_map is None:
        forward_map = {}
    if len(back_bounds):
        if len(back_bounds) == 1:
            C = back_bounds[0]
            bias = K.cast(0.0, model.layers[0].dtype) * C[:, 0]
            back_bounds = [C, bias] * 2
    result = crown_model(
        model,
        input_tensors,
        back_bounds,
        input_dim,
        IBP,
        forward,
        convex_domain,
        finetune,
        forward_map,
        softmax_to_linear,
        joint,
        layer_fn,
        fuse=True,
        **kwargs,
    )

    input_tensors, output, backward_map, toto = result

    output = convert_2_mode(
        mode_from=get_mode(IBP, forward),
        mode_to=get_mode(final_ibp, final_forward),
        convex_domain=convex_domain,
        dtype=model.layers[0].dtype,
    )(output)
    return input_tensors, output, backward_map, toto


# Aliasing
convert_backward_model = get_backward_model
