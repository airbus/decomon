from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.backend import observe_object_name
from keras.engine.node import Node
from tensorflow.keras.layers import (
    Activation,
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
from decomon.layers.utils import is_a_merge_layer
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


def has_merge_layers(model: Model) -> bool:
    return any(is_a_merge_layer(layer) for layer in model.layers)


def get_input_tensors(
    model: Model,
    convex_domain: Dict[str, Any],
    ibp: bool = True,
    affine: bool = True,
) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    input_dim = get_input_dim(model)
    input_shape = None
    input_shape_vec = None

    for input_layer in model._input_layers:
        if len(input_layer.input_shape) > 1:
            raise ValueError(f"Expected one input tensor but got {len(input_layer.input_shape)}")
        input_shape_vec_ = input_layer.input_shape[0]
        input_shape_ = tuple(list(input_shape_vec_)[1:])

        if input_shape_vec is None:
            input_shape_vec = input_shape_vec_
        if input_shape is None:
            input_shape = input_shape_
        else:
            if not np.allclose(input_shape, input_shape_):
                raise ValueError("Expected that every input layers use the same input_tensor")

    input_shape_x: Tuple[int, ...]
    if len(convex_domain) == 0 or convex_domain["name"] != ConvexDomainType.BALL:
        input_shape_x = (2, input_dim)
    else:
        input_shape_x = (input_dim,)

    z_tensor = Input(shape=input_shape_x, dtype=model.layers[0].dtype)

    if len(convex_domain) == 0 or convex_domain["name"] != ConvexDomainType.BALL:

        if ibp:
            u_c_tensor = Lambda(lambda z: z[:, 1], dtype=z_tensor.dtype)(z_tensor)
            l_c_tensor = Lambda(lambda z: z[:, 0], dtype=z_tensor.dtype)(z_tensor)

        if affine:
            z_value = K.cast(0.0, z_tensor.dtype)
            o_value = K.cast(1.0, z_tensor.dtype)
            W = Lambda(lambda z: tf.linalg.diag(z_value * z[:, 0] + o_value), dtype=z_tensor.dtype)(z_tensor)
            b = Lambda(lambda z: z_value * z[:, 1], dtype=z_tensor.dtype)(z_tensor)

    else:

        if convex_domain["p"] == np.inf:
            radius = convex_domain["eps"]
            if ibp:
                u_c_tensor = Lambda(
                    lambda var: var + K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
                )(z_tensor)
                l_c_tensor = Lambda(
                    lambda var: var - K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
                )(z_tensor)
            if affine:
                z_value = K.cast(0.0, model.layers[0].dtype)
                o_value = K.cast(1.0, model.layers[0].dtype)

                W = tf.linalg.diag(z_value * u_c_tensor + o_value)
                b = z_value * u_c_tensor

        else:
            z_value = K.cast(0.0, model.layers[0].dtype)
            o_value = K.cast(1.0, model.layers[0].dtype)

            def get_bounds(z: tf.Tensor) -> List[tf.Tensor]:
                output = []
                if affine:
                    W = tf.linalg.diag(z_value * z + o_value)
                    b = z_value * z
                    output += [W, b]
                if ibp:
                    u_c_ = get_upper(z, W, b, convex_domain)
                    l_c_ = get_lower(z, W, b, convex_domain)
                    output += [u_c_, l_c_]
                return output

            output_ = get_bounds(z_tensor)
            if ibp:
                u_c_tensor, l_c_tensor = output_[-2:]
            if affine:
                W, b = output_[:2]

    if ibp and affine:
        input_tensors = [z_tensor] + [u_c_tensor, W, b] + [l_c_tensor, W, b]
    elif ibp and not affine:
        input_tensors = [u_c_tensor, l_c_tensor]
    elif not ibp and affine:
        input_tensors = [z_tensor] + [W, b] + [W, b]
    else:
        raise NotImplementedError("not ibp and not affine not implemented")

    return z_tensor, input_tensors


def get_input_tensors_keras_only(
    model: Model,
    input_shape: Tuple[int, ...],
) -> List[tf.Tensor]:
    input_tensors = []
    for i in range(len(model._input_layers)):
        input_tensors.append(Input(input_shape[1:], dtype=model.layers[0].dtype))
    return input_tensors


def get_input_dim(layer: Layer) -> int:
    if isinstance(layer.input_shape, list):
        return np.prod(layer.input_shape[0][1:])
    else:
        return np.prod(layer.input_shape[1:])


def prepare_inputs_for_layer(
    inputs: Union[Tuple[tf.Tensor, ...], List[tf.Tensor], tf.Tensor]
) -> Union[Tuple[tf.Tensor, ...], List[tf.Tensor], tf.Tensor]:
    """Prepare inputs for keras/decomon layers.

    Some Keras layers do not like list of tensors even with one single tensor.
    So we keep only the tensor in this case.

    """
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) == 1:
            return inputs[0]
    return inputs


def wrap_outputs_from_layer_in_list(
    outputs: Union[Tuple[tf.Tensor, ...], List[tf.Tensor], tf.Tensor]
) -> List[tf.Tensor]:
    if not isinstance(outputs, list):
        if isinstance(outputs, tuple):
            return list(outputs)
        else:
            return [outputs]
    else:
        return outputs


def split_activation(layer: Layer) -> List[Layer]:
    config = layer.get_config()
    activation = config.pop("activation", None)
    if activation is None or activation == "linear" or isinstance(layer, Activation):
        return [layer]
    else:
        # layer without activation
        config["name"] = f"{layer.name}_wo_activation"
        layer_wo_activation = layer.__class__.from_config(config)
        # build the layer
        if not hasattr(layer, "input_shape"):
            raise RuntimeError("The layer should properly initialized so that layer.input_shape is defined.")
        inputs = Input(type_spec=layer.input.type_spec)
        outputs = layer_wo_activation(inputs)
        # use same weights
        actual_activation = layer.activation  # store activation
        layer.activation = None  # remove activation in case of a layer having weights (would be in layer.get_weights())
        layer_wo_activation.set_weights(layer.get_weights())
        layer.activation = actual_activation  # put back the actual activation
        # even share the object themselves if possible
        # (dense and conv2d have kernel and bias weight attributes)
        if hasattr(layer_wo_activation, "kernel"):
            layer_wo_activation.kernel = layer.kernel
        if hasattr(layer_wo_activation, "bias"):
            layer_wo_activation.bias = layer.bias
        # activation layer
        if isinstance(activation, dict):
            if isinstance(layer.activation, Layer):  # can be an Activation, a PReLU, or a deel.lip.activations layer
                activation_layer = layer.activation
                # update the name to starts with main layer name
                activation_layer_name = f"{layer.name}_activation_{layer.activation.name}"
                observe_object_name(activation_layer_name)
                activation_layer._name = activation_layer_name
            else:
                raise RuntimeError("Cannot construct activation layer from layer.activation!")
        else:
            activation_layer = Activation(activation=activation, dtype=layer.dtype, name=f"{layer.name}_activation")
        # build activation layer
        activation_layer(outputs)
        return [layer_wo_activation, activation_layer]


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
