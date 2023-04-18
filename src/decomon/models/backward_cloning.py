from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.engine.node import Node
from tensorflow.keras.layers import Add, Average, Concatenate
from tensorflow.keras.layers import Conv2D as Conv
from tensorflow.keras.layers import Dense, Dropout, Lambda, Layer, Permute, Reshape
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import has_arg, to_list

from decomon.backward_layers.backward_layers import to_backward
from decomon.backward_layers.backward_merge import BackwardMerge
from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.crown import (
    Convert2BackwardMode,
    Convert2Mode,
    Fuse,
    MergeWithPrevious,
)
from decomon.backward_layers.utils import merge_with_previous
from decomon.layers.core import ForwardMode
from decomon.layers.utils import softmax_to_linear as softmax_2_linear
from decomon.models.forward_cloning import OutputMapDict
from decomon.models.utils import get_depth_dict, get_mode
from decomon.utils import Slope, get_lower, get_upper


def get_disconnected_input(
    mode: Union[str, ForwardMode], convex_domain: Optional[Dict[str, Any]], dtype: Union[str, tf.DType] = K.floatx()
) -> Layer:
    mode = ForwardMode(mode)

    def disco_priv(inputs_: List[tf.Tensor]) -> List[tf.Tensor]:

        if mode == ForwardMode.IBP:
            return inputs_
        elif mode == ForwardMode.AFFINE:
            x_0, w_f_u, b_f_u, w_f_l, b_f_l = inputs_
            u_c = get_upper(x_0, w_f_u, b_f_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_f_l, b_f_l, convex_domain=convex_domain)

        elif mode == ForwardMode.HYBRID:
            _, u_c, _, _, l_c, _, _ = inputs_
        else:
            raise ValueError("Unknown mode.")

        x_0 = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
        w_u_ = tf.linalg.diag(K.cast(0.0, x_0.dtype) * u_c + K.cast(1.0, x_0.dtype))
        b_u_ = K.cast(0.0, x_0.dtype) * u_c
        # w_u_ = tf.linalg.diag(K.cast(0., x_0.dtype)*u_c)

        if mode == ForwardMode.AFFINE:
            return [x_0, w_u_, b_u_, w_u_, b_u_]
        if mode == ForwardMode.HYBRID:
            return [x_0, u_c, w_u_, b_u_, l_c, w_u_, b_u_]
        else:
            raise ValueError("Unknown mode.")

    return Lambda(disco_priv, dtype=dtype)


def retrieve_layer(
    node: Node,
    layer_fn: Callable[[Layer], BackwardLayer],
    backward_map: Dict[int, BackwardLayer],
    joint: bool = True,
) -> BackwardLayer:
    if id(node) in backward_map:
        backward_layer = backward_map[id(node)]
    else:
        backward_layer = layer_fn(node.outbound_layer)
        if joint:
            backward_map[id(node)] = backward_layer
    return backward_layer


def crown_(
    node: Node,
    ibp: bool,
    affine: bool,
    convex_domain: Optional[Dict[str, Any]],
    input_map: Dict[int, List[tf.Tensor]],
    layer_fn: Callable[[Layer], BackwardLayer],
    backward_bounds: List[tf.Tensor],
    backward_map: Optional[Dict[int, BackwardLayer]] = None,
    joint: bool = True,
    fuse: bool = True,
    output_map: Optional[Dict[int, List[tf.Tensor]]] = None,
    merge_layers: Optional[Layer] = None,
    fuse_layer: Optional[Layer] = None,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], Optional[Layer]]:
    """


    :param node:
    :param ibp:
    :param affine:
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

    if isinstance(node.outbound_layer, Model):

        inputs_ = get_disconnected_input(get_mode(ibp, affine), convex_domain, dtype=inputs[0].dtype)(inputs)
        _, backward_bounds_, _, _ = crown_model(
            model=node.outbound_layer,
            input_tensors=inputs_,
            backward_bounds=backward_bounds,
            ibp=ibp,
            affine=affine,
            convex_domain=None,
            finetune=False,
            joint=joint,
            fuse=False,
            **kwargs,
        )

    else:
        backward_layer = retrieve_layer(node=node, layer_fn=layer_fn, backward_map=backward_map, joint=joint)

        if id(node) not in output_map:
            backward_bounds_ = backward_layer(inputs)
            output_map[id(node)] = backward_bounds_
        else:
            backward_bounds_ = output_map[id(node)]

        if not isinstance(backward_bounds_, list):
            backward_bounds_ = [e for e in backward_bounds_]
            # import pdb; pdb.set_trace()
        if len(backward_bounds):
            if merge_layers is None:
                merge_layers = MergeWithPrevious(backward_bounds_[0].shape, backward_bounds[0].shape)
            backward_tmp = merge_layers(backward_bounds_ + backward_bounds)
            backward_bounds_ = backward_tmp

    if not isinstance(backward_bounds_, list):
        backward_bounds = [e for e in backward_bounds_]
    else:
        backward_bounds = backward_bounds_

    parents = node.parent_nodes

    if len(parents):

        if len(parents) > 1:
            if isinstance(backward_layer, BackwardMerge):
                raise NotImplementedError()
                crown_bound_list = []
                for (backward_bound, parent) in zip(backward_bounds, parents):

                    crown_bound_i, _ = crown_(
                        node=parent,
                        ibp=ibp,
                        affine=affine,
                        convex_domain=convex_domain,
                        input_map=input_map,
                        layer_fn=layer_fn,
                        backward_bounds=backward_bound,
                        backward_map=backward_map,
                        joint=joint,
                        fuse=fuse,
                    )

                    crown_bound_list.append(crown_bound_i)

                # import pdb; pdb.set_trace()
                avg_layer = Average(dtype=node.outbound_layer.dtype)

                # crown_bound = [avg_layer([e[i] for e in crown_bound_list]) for i in range(4)]
                crown_bound = crown_bound_list[0]

            else:
                raise NotImplementedError()
        else:
            crown_bound, fuse_layer_ = crown_(
                node=parents[0],
                ibp=ibp,
                affine=affine,
                convex_domain=convex_domain,
                input_map=input_map,
                layer_fn=layer_fn,
                backward_bounds=backward_bounds,
                backward_map=backward_map,
                joint=joint,
                fuse=fuse,
                output_map=output_map,
                merge_layers=None,  # AKA merge_layers
                fuse_layer=fuse_layer,
            )
            if fuse_layer is None:
                fuse_layer = fuse_layer_
        return crown_bound, fuse_layer
    else:
        # do something
        if fuse:
            if fuse_layer is None:
                fuse_layer = Fuse(get_mode(ibp=ibp, affine=affine))
            result = fuse_layer(inputs + backward_bounds)

            return result, fuse_layer

        else:
            return backward_bounds, fuse_layer


def get_input_nodes(
    model: Model,
    dico_nodes: Dict[int, List[Node]],
    ibp: bool,
    affine: bool,
    input_tensors: List[tf.Tensor],
    output_map: OutputMapDict,
    layer_fn: Callable[[Layer], BackwardLayer],
    joint: bool,
    set_mode_layer: Layer,
    convex_domain: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[int, List[tf.Tensor]], Dict[int, BackwardLayer], Dict[int, List[tf.Tensor]]]:

    keys = [e for e in dico_nodes.keys()]
    keys.sort(reverse=True)
    fuse_layer = None
    input_map: Dict[int, List[tf.Tensor]] = {}
    backward_map: Dict[int, BackwardLayer] = {}
    if convex_domain is None:
        convex_domain = {}
    crown_map: Dict[int, List[tf.Tensor]] = {}
    for depth in keys:
        nodes = dico_nodes[depth]
        for node in nodes:
            layer = node.outbound_layer

            parents = node.parent_nodes
            if not len(parents):
                # if 'debug' in kwargs.keys():
                #    import pdb; pdb.set_trace()
                input_map[id(node)] = input_tensors
            else:
                output: List[tf.Tensor] = []
                for parent in parents:
                    # do something
                    if id(parent) in output_map.keys():
                        output += output_map[id(parent)]
                    else:
                        output_crown, fuse_layer_ = crown_(
                            node=parent,
                            ibp=ibp,
                            affine=affine,
                            input_map=input_map,
                            layer_fn=layer_fn,
                            backward_bounds=[],
                            backward_map=backward_map,
                            joint=joint,
                            fuse=True,
                            convex_domain=convex_domain,
                            output_map=crown_map,
                            merge_layers=None,  # AKA merge_layers
                            fuse_layer=fuse_layer,
                        )
                        if fuse_layer is None:
                            fuse_layer = fuse_layer_

                        # convert output_crown in the right mode
                        if set_mode_layer is None:
                            set_mode_layer = Convert2BackwardMode(get_mode(ibp, affine), convex_domain)
                        output_crown_ = set_mode_layer(input_tensors + output_crown)
                        output += to_list(output_crown_)
                        # crown_map[id(parent)]=output_crown_

                input_map[id(node)] = output
    return input_map, backward_map, crown_map


def crown_model(
    model: Model,
    input_tensors: List[tf.Tensor],
    back_bounds: Optional[List[tf.Tensor]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    ibp: bool = True,
    affine: bool = True,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    softmax_to_linear: bool = True,
    joint: bool = True,
    layer_fn: Callable[..., BackwardLayer] = to_backward,
    fuse: bool = True,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], Dict[int, BackwardLayer], None]:
    if back_bounds is None:
        back_bounds = []
    if forward_map is None:
        forward_map = {}
    if not isinstance(model, Model):
        raise ValueError()
    # import time
    # zero_time = time.process_time()
    has_softmax = False
    if softmax_to_linear:
        model, has_softmax = softmax_2_linear(model)  # do better because you modify the model eventually

    # layer_fn
    ##########
    has_iter = False
    if layer_fn is not None and len(layer_fn.__code__.co_varnames) == 1 and "layer" in layer_fn.__code__.co_varnames:
        has_iter = True

    if not has_iter:
        layer_fn_copy = deepcopy(layer_fn)

        def func(layer: Layer) -> Layer:
            return layer_fn_copy(
                layer, mode=get_mode(ibp, affine), finetune=finetune, convex_domain=convex_domain, slope=slope
            )

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
    set_mode_layer = Convert2BackwardMode(get_mode(ibp, affine), convex_domain)

    input_map, backward_map, crown_map = get_input_nodes(
        model=model,
        dico_nodes=dico_nodes,
        ibp=ibp,
        affine=affine,
        input_tensors=input_tensors,
        output_map=forward_map,
        layer_fn=layer_fn,
        joint=joint,
        convex_domain=convex_domain,
        set_mode_layer=set_mode_layer,
        **kwargs,
    )
    # time_1 = time.process_time()
    # print('step1', time_1-zero_time)
    # retrieve output nodes
    output = []
    output_nodes = dico_nodes[0]
    # the ordering may change
    output_names = [tensor._keras_history.layer.name for tensor in to_list(model.output)]
    fuse_layer = None
    for output_name in output_names:
        for node in output_nodes:
            if node.outbound_layer.name == output_name:
                # compute with crown
                output_crown, fuse_layer = crown_(
                    node=node,
                    ibp=ibp,
                    affine=affine,
                    input_map=input_map,
                    layer_fn=layer_fn,
                    backward_bounds=back_bounds,
                    backward_map=backward_map,
                    joint=joint,
                    fuse=fuse,
                    convex_domain=convex_domain,
                    output_map=crown_map,
                    fuse_layer=fuse_layer,
                )
                # time_2 = time.process_time()
                # print('step2', time_2-time_1)
                if fuse:
                    # import pdb; pdb.set_trace()
                    output += to_list(set_mode_layer(input_tensors + output_crown))
                else:
                    output = output_crown

    return input_tensors, output, backward_map, None


def convert_backward(
    model: Model,
    input_tensors: List[tf.Tensor],
    back_bounds: Optional[List[tf.Tensor]] = None,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    ibp: bool = True,
    affine: bool = True,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    forward_map: Optional[OutputMapDict] = None,
    softmax_to_linear: bool = True,
    joint: bool = True,
    layer_fn: Callable[..., BackwardLayer] = to_backward,
    final_ibp: bool = True,
    final_affine: bool = False,
    **kwargs: Any,
) -> Tuple[List[tf.Tensor], List[tf.Tensor], Dict[int, BackwardLayer], None]:
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
        model=model,
        input_tensors=input_tensors,
        back_bounds=back_bounds,
        slope=slope,
        ibp=ibp,
        affine=affine,
        convex_domain=convex_domain,
        finetune=finetune,
        forward_map=forward_map,
        softmax_to_linear=softmax_to_linear,
        joint=joint,
        layer_fn=layer_fn,
        fuse=True,
        **kwargs,
    )

    input_tensors, output, backward_map, toto = result
    mode_from = get_mode(ibp, affine)
    mode_to = get_mode(final_ibp, final_affine)
    output = Convert2Mode(
        mode_from=mode_from,
        mode_to=mode_to,
        convex_domain=convex_domain,
        dtype=model.layers[0].dtype,
    )(output)
    if mode_to != mode_from and mode_from == ForwardMode.IBP:

        f_input = Lambda(lambda z: Concatenate(1)([z[0][:, None], z[1][:, None]]))
        output[0] = f_input([input_tensors[1], input_tensors[0]])
    return input_tensors, output, backward_map, toto
