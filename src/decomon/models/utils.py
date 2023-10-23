from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import (
    Activation,
    Concatenate,
    Flatten,
    Input,
    Lambda,
    Layer,
    Maximum,
    Minimum,
)
from keras.src import Functional
from keras.src.ops.node import Node

from decomon.core import (
    BallDomain,
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_mode,
)
from decomon.layers.utils import is_a_merge_layer

try:
    from deel.lip.layers import LipschitzLayer
except ImportError:
    LipschitzLayer = type(None)


class ConvertMethod(str, Enum):
    CROWN = "crown"
    CROWN_FORWARD_IBP = "crown-forward-ibp"
    CROWN_FORWARD_AFFINE = "crown-forward-affine"
    CROWN_FORWARD_HYBRID = "crown-forward-hybrid"
    FORWARD_IBP = "forward-ibp"
    FORWARD_AFFINE = "forward-affine"
    FORWARD_HYBRID = "forward-hybrid"


def get_ibp_affine_from_method(method: Union[str, ConvertMethod]) -> Tuple[bool, bool]:
    method = ConvertMethod(method)
    if method in [ConvertMethod.FORWARD_IBP, ConvertMethod.CROWN_FORWARD_IBP]:
        return True, False
    if method in [ConvertMethod.FORWARD_AFFINE, ConvertMethod.CROWN_FORWARD_AFFINE]:
        return False, True
    if method in [ConvertMethod.FORWARD_HYBRID, ConvertMethod.CROWN_FORWARD_HYBRID]:
        return True, True
    if method == ConvertMethod.CROWN:
        return True, False
    return True, True


class FeedDirection(str, Enum):
    FORWARD = "feed_forward"
    BACKWARD = "feed_backward"


def get_direction(method: Union[str, ConvertMethod]) -> FeedDirection:
    if ConvertMethod(method) in [ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_AFFINE, ConvertMethod.FORWARD_HYBRID]:
        return FeedDirection.FORWARD
    else:
        return FeedDirection.BACKWARD


def has_merge_layers(model: Model) -> bool:
    return any(is_a_merge_layer(layer) for layer in model.layers)


def check_model2convert_inputs(model: Model) -> None:
    """Check that the model to convert satisfy the hypotheses for decomon on inputs.

    Which means:

    - only one input
    - the input must be flattened: only batchsize + another dimension

    """
    if len(model.inputs) > 1:
        raise ValueError("The model must have only 1 input to be converted.")
    if len(model.inputs[0].shape) > 2:
        raise ValueError("The model must have a flattened input to be converted.")


def get_input_tensors(
    model: Model,
    perturbation_domain: PerturbationDomain,
    ibp: bool = True,
    affine: bool = True,
) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
    input_dim = get_input_dim(model)
    mode = get_mode(ibp=ibp, affine=affine)
    dc_decomp = False
    inputs_outputs_spec = InputsOutputsSpec(dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain)
    empty_tensor = inputs_outputs_spec.get_empty_tensor()

    input_shape_x = perturbation_domain.get_x_input_shape(input_dim)
    z_tensor = Input(shape=input_shape_x, dtype=model.layers[0].dtype)
    u_c_tensor, l_c_tensor, W, b, h, g = (
        empty_tensor,
        empty_tensor,
        empty_tensor,
        empty_tensor,
        empty_tensor,
        empty_tensor,
    )

    if isinstance(perturbation_domain, BoxDomain):
        if ibp:
            u_c_tensor = Lambda(lambda z: z[:, 1], dtype=z_tensor.dtype)(z_tensor)
            l_c_tensor = Lambda(lambda z: z[:, 0], dtype=z_tensor.dtype)(z_tensor)

        if affine:
            z_value = K.cast(0.0, z_tensor.dtype)
            o_value = K.cast(1.0, z_tensor.dtype)
            W = Lambda(lambda z: tf.linalg.diag(z_value * z[:, 0] + o_value), dtype=z_tensor.dtype)(z_tensor)
            b = Lambda(lambda z: z_value * z[:, 1], dtype=z_tensor.dtype)(z_tensor)

    elif isinstance(perturbation_domain, BallDomain):
        if perturbation_domain.p == np.inf:
            radius = perturbation_domain.eps
            u_c_tensor = Lambda(
                lambda var: var + K.cast(radius, dtype=model.layers[0].dtype), dtype=model.layers[0].dtype
            )(z_tensor)
            if ibp:
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

            def get_bounds(z: keras.KerasTensor) -> List[keras.KerasTensor]:
                outputs = []
                W = tf.linalg.diag(z_value * z + o_value)
                b = z_value * z
                if affine:
                    outputs += [W, b]
                if ibp:
                    u_c_out = perturbation_domain.get_upper(z, W, b)
                    l_c_out = perturbation_domain.get_lower(z, W, b)
                    outputs += [u_c_out, l_c_out]
                return outputs

            outputs = get_bounds(z_tensor)
            if ibp:
                u_c_tensor, l_c_tensor = outputs[-2:]
            if affine:
                W, b = outputs[:2]
    else:
        raise NotImplementedError(f"Not implemented for perturbation domain type {type(perturbation_domain)}")

    input_tensors = inputs_outputs_spec.extract_inputsformode_from_fullinputs(
        [z_tensor, u_c_tensor, W, b, l_c_tensor, W, b, h, g]
    )

    return z_tensor, input_tensors


def get_input_dim(layer: Layer) -> int:
    if isinstance(layer.input, list):
        if len(layer.input) == 0:
            return 0
        return int(np.prod(layer.input[0].shape[1:]))
    else:
        return int(np.prod(layer.input.shape[1:]))


def prepare_inputs_for_layer(
    inputs: Union[Tuple[keras.KerasTensor, ...], List[keras.KerasTensor], keras.KerasTensor]
) -> Union[Tuple[keras.KerasTensor, ...], List[keras.KerasTensor], keras.KerasTensor]:
    """Prepare inputs for keras/decomon layers.

    Some Keras layers do not like list of tensors even with one single tensor.
    So we keep only the tensor in this case.

    """
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) == 1:
            return inputs[0]
    return inputs


def wrap_outputs_from_layer_in_list(
    outputs: Union[Tuple[keras.KerasTensor, ...], List[keras.KerasTensor], keras.KerasTensor]
) -> List[keras.KerasTensor]:
    if not isinstance(outputs, list):
        if isinstance(outputs, tuple):
            return list(outputs)
        else:
            return [outputs]
    else:
        return outputs


def _share_no_attributes(new_layer: Layer, old_layer: Layer) -> None:
    return


def split_activation(
    layer: Layer, share_some_attributes: Optional[Callable[[Layer, Layer], None]] = None
) -> List[Layer]:
    # init
    if share_some_attributes is None:
        share_some_attributes = _share_no_attributes
    # get activation
    config = layer.get_config()
    activation = config.pop("activation", None)
    # split if needed
    if activation is None or activation == "linear" or isinstance(layer, Activation):
        return [layer]
    else:
        # layer without activation
        config["name"] = f"{layer.name}_wo_activation"
        layer_wo_activation = layer.__class__.from_config(config)
        share_some_attributes(layer_wo_activation, layer)  # share (deel-lip) attributes
        # build the layer
        if not hasattr(layer, "input_shape"):
            raise RuntimeError("The layer should properly initialized so that layer.input_shape is defined.")
        inputs = Input(
            shape=layer.input.shape,
            dtype=layer.input.dtype,
        )
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
                activation_layer._name = activation_layer_name
            else:
                raise RuntimeError("Cannot construct activation layer from layer.activation!")
        else:
            activation_layer = Activation(activation=activation, dtype=layer.dtype, name=f"{layer.name}_activation")
        # build activation layer
        activation_layer(outputs)
        return [layer_wo_activation, activation_layer]


def convert_deellip_to_keras(layer: Layer) -> Layer:
    # init deel-lip attributes (keep exisiting ones)
    share_deellip_attributes(layer, layer)
    # update is_lipschitz
    if isinstance(layer, LipschitzLayer) or hasattr(layer, "vanilla_export"):
        layer.is_lipschitz = True
    if hasattr(layer, "vanilla_export"):
        if not hasattr(layer, "input_shape"):
            raise RuntimeError("The layer should properly initialized so that layer.input_shape is defined.")
        new_layer = layer.vanilla_export()
        # build layer
        inputs = Input(
            shape=layer.input.shape,
            dtype=layer.input.dtype,
        )
        new_layer(inputs)
        # share deel-lip attributes of original layer
        share_deellip_attributes(new_layer, layer)
        layer = new_layer

    return layer


def share_deellip_attributes(new_layer: Layer, old_layer: Layer = None) -> None:
    new_layer.is_lipschitz = getattr(old_layer, "is_lipschitz", False)
    new_layer.deellip_classname = getattr(old_layer, "deellip_classname", new_layer.__class__.__name__)
    new_layer.k_coef_lip = getattr(old_layer, "k_coef_lip", -1.0)


def preprocess_layer(layer: Layer) -> List[Layer]:
    layer = convert_deellip_to_keras(layer)
    layers = split_activation(layer, share_some_attributes=share_deellip_attributes)
    # convert activation layers (if were embedded deel-lip layers)
    return [convert_deellip_to_keras(l) for l in layers]


def is_input_node(node: Node) -> bool:
    return len(node.input_tensors) == 0


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


class Convert2Mode(Layer):
    def __init__(
        self,
        mode_from: Union[str, ForwardMode],
        mode_to: Union[str, ForwardMode],
        perturbation_domain: PerturbationDomain,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.mode_from = ForwardMode(mode_from)
        self.mode_to = ForwardMode(mode_to)
        self.perturbation_domain = perturbation_domain

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        mode_from = self.mode_from
        mode_to = self.mode_to
        perturbation_domain = self.perturbation_domain

        dc_decomp = False
        inputs_outputs_spec_from = InputsOutputsSpec(
            dc_decomp=dc_decomp, mode=mode_from, perturbation_domain=perturbation_domain
        )
        inputs_outputs_spec_to = InputsOutputsSpec(
            dc_decomp=dc_decomp, mode=mode_to, perturbation_domain=perturbation_domain
        )

        compute_ibp_from_affine = mode_from == ForwardMode.AFFINE and mode_to != ForwardMode.AFFINE
        tight = mode_from == ForwardMode.HYBRID and mode_to != ForwardMode.AFFINE
        compute_dummy_affine = mode_from == ForwardMode.IBP and mode_to != ForwardMode.IBP
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs_outputs_spec_from.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=compute_ibp_from_affine, tight=tight
        )
        dtype = x.dtype

        if compute_dummy_affine:
            x = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            z_value = K.cast(0.0, dtype=dtype)
            w = tf.linalg.diag(z_value * l_c)
            w_u = w
            b_u = u_c
            w_l = w
            b_l = l_c

        return inputs_outputs_spec_to.extract_outputsformode_from_fulloutputs([x, u_c, w_u, b_u, l_c, w_l, b_l, h, g])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"mode_from": self.mode_from, "mode_to": self.mode_to, "perturbation_domain": self.perturbation_domain}
        )
        return config


def ensure_functional_model(model: Model) -> Functional:
    if isinstance(model, Functional):
        return model
    elif isinstance(model, Sequential):
        model = Model(model.inputs, model.outputs)
        assert isinstance(model, Functional)  # should be the case after passage in Model.__init__()
        return model
    else:
        raise NotImplementedError("Decomon model available only for functional or sequential models.")
