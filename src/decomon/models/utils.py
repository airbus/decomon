from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K
from keras.engine.node import Node
from tensorflow.keras.layers import (
    Concatenate,
    Flatten,
    Input,
    Lambda,
    Layer,
    Maximum,
    Minimum,
)
from tensorflow.keras.models import Model

from decomon.layers.core import ForwardMode
from decomon.utils import (
    ConvexDomainType,
    get_lower,
    get_lower_layer,
    get_upper,
    get_upper_layer,
)


class ConvertMethod(Enum):
    CROWN = "crown"
    CROWN_FORWARD_IBP = "crown-forward-ibp"
    CROWN_FORWARD_AFFINE = "crown-forward-affine"
    CROWN_FORWARD_HYBRID = "crown-forward-hybrid"
    FORWARD_IBP = "forward-ibp"
    FORWARD_AFFINE = "forward-affine"
    FORWARD_HYBRID = "forward-hybrid"


def get_input_tensors(
    model: Model,
    input_dim: int,
    convex_domain: Optional[Dict[str, Any]] = None,
    ibp: bool = True,
    affine: bool = True,
) -> List[tf.Tensor]:
    input_tensors = []
    for i in range(len(model._input_layers)):
        tmp = check_input_tensors_sequential(model, None, input_dim, input_dim, ibp, affine, False, convex_domain)
        input_tensors += tmp
    return input_tensors


def get_input_dim(model: Model) -> int:
    if isinstance(model.input_shape, list):
        return np.prod(model.input_shape[0][1:])
    else:
        return np.prod(model.input_shape[1:])


def check_input_tensors_sequential(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    ibp: bool,
    affine: bool,
    dc_decomp: bool,
    convex_domain: Optional[Dict[str, Any]],
) -> List[tf.Tensor]:

    if convex_domain is None:
        convex_domain = {}
    if input_tensors is None:  # no predefined input tensors

        input_shape = list(K.int_shape(model.input)[1:])

        if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
            input_dim_ = (2, input_dim)
            z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
        elif convex_domain["name"] == ConvexDomainType.BOX and not isinstance(input_dim, tuple):
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
        if affine:
            b_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            b_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if input_dim_init > 0:
                w_u_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple([input_dim] + input_shape), dtype=model.layers[0].dtype)
            else:
                w_u_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
                w_l_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)

        if ibp:
            u_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            l_c_tensor = Input(tuple(input_shape), dtype=model.layers[0].dtype)
            if affine:  # hybrid mode
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
                # only ibp
                input_tensors = [
                    u_c_tensor,
                    l_c_tensor,
                ]
        elif affine:
            # affine mode
            input_tensors = [
                z_tensor,
                w_u_tensor,
                b_u_tensor,
                w_l_tensor,
                b_l_tensor,
            ]
        else:
            raise NotImplementedError("not ibp and not affine not implemented")

        if dc_decomp:
            input_tensors += [h_tensor, g_tensor]

    else:
        # assert that input_tensors is a List of 6 InputLayer objects
        # If input tensors are provided, the original model's InputLayer is
        # overwritten with a different InputLayer.
        assert isinstance(input_tensors, list), "expected input_tensors to be a List or None, but got {}".format(
            type(input_tensors)
        )

        if dc_decomp:
            if ibp and affine:
                assert len(input_tensors) == 9, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if ibp and not affine:
                assert len(input_tensors) == 4, "wrong number of inputs, expexted 6 but got {}".format(
                    len(input_tensors)
                )
            if not ibp and affine:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 8 but got {}".format(
                    len(input_tensors)
                )
        else:
            if ibp and affine:
                assert len(input_tensors) == 7, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if ibp and not affine:
                assert len(input_tensors) == 2, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )
            if not ibp and affine:
                assert len(input_tensors) == 5, "wrong number of inputs, expexted 10 but got {}".format(
                    len(input_tensors)
                )

    return input_tensors


def get_input_tensor_x(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    convex_domain: Dict[str, Any],
) -> Input:
    if len(convex_domain) == 0 and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    elif convex_domain["name"] == ConvexDomainType.BOX and not isinstance(input_dim, tuple):
        input_dim_ = (2, input_dim)
        z_tensor = Input(input_dim_, dtype=model.layers[0].dtype)
    else:

        if isinstance(input_dim, tuple):
            z_tensor = Input(input_dim, dtype=model.layers[0].dtype)
        else:
            z_tensor = Input((input_dim,), dtype=model.layers[0].dtype)
    return z_tensor


def check_input_tensors_functionnal(
    model: Model,
    input_tensors: Optional[List[tf.Tensor]],
    input_dim: int,
    input_dim_init: int,
    ibp: bool,
    affine: bool,
    dc_decomp: bool,
    convex_domain: Optional[Dict[str, Any]],
) -> List[tf.Tensor]:

    raise NotImplementedError()


def get_mode(ibp: bool = True, affine: bool = True) -> ForwardMode:

    if ibp:
        if affine:
            return ForwardMode.HYBRID
        else:
            return ForwardMode.IBP
    else:
        return ForwardMode.AFFINE


def get_depth_dict(model: Model) -> Dict[int, List[Node]]:

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    nodes_list = []

    dico_depth: Dict[int, int] = {}
    dico_nodes: Dict[int, List[Node]] = {}

    def fill_dico(node: Node, dico_depth: Optional[Dict[int, int]] = None) -> Dict[int, int]:
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


def get_inner_layers(model: Model) -> int:

    count = 0
    for layer in model.layers:
        if isinstance(layer, Model):
            count += get_inner_layers(layer)
        else:
            count += 1
    return count
