from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
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
from decomon.keras_utils import BatchedIdentityLike, share_weights_and_build
from decomon.layers.utils import is_a_merge_layer
from decomon.types import BackendTensor


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

    input_shape_x = perturbation_domain.get_x_input_shape_wo_batchsize(input_dim)
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
            W = BatchedIdentityLike()(z_tensor[:, 0])
            b = K.zeros_like(z_tensor[:, 0])

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
                W = BatchedIdentityLike()(u_c_tensor)
                b = K.zeros_like(u_c_tensor)

        else:
            W = BatchedIdentityLike()(z_tensor)
            b = K.zeros_like(z_tensor)
            if ibp:
                u_c_tensor = perturbation_domain.get_upper(z_tensor, W, b)
                l_c_tensor = perturbation_domain.get_lower(z_tensor, W, b)

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


def split_activation(layer: Layer) -> List[Layer]:
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
        # get expected weights names without activation (activation layer could add some weights)
        layer_wo_activation_tmp = layer.__class__.from_config(config)  # create a temporary layer w/o activation
        layer_wo_activation_tmp(layer.input)  # build it
        weights_names = [w.name for w in layer_wo_activation_tmp.weights]
        # share weights (and build the new layer)
        share_weights_and_build(original_layer=layer, new_layer=layer_wo_activation, weight_names=weights_names)

        # activation layer
        if isinstance(activation, dict):
            if isinstance(layer.activation, Layer):  # can be an Activation or a PReLU layer
                activation_layer = layer.activation
                # update the name to starts with main layer name
                activation_layer_name = f"{layer.name}_activation_{layer.activation.name}"
                activation_layer.name = activation_layer_name
            else:
                raise RuntimeError("Cannot construct activation layer from layer.activation!")
        else:
            activation_layer = Activation(activation=activation, dtype=layer.dtype, name=f"{layer.name}_activation")
        # build activation layer
        activation_layer(layer_wo_activation.output)
        return [layer_wo_activation, activation_layer]


def preprocess_layer(layer: Layer) -> List[Layer]:
    return split_activation(layer)


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
        input_dim: int = -1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.mode_from = ForwardMode(mode_from)
        self.mode_to = ForwardMode(mode_to)
        self.perturbation_domain = perturbation_domain
        self.input_dim = input_dim
        dc_decomp = False
        self.dc_decomp = dc_decomp
        self.inputs_outputs_spec_from = InputsOutputsSpec(
            dc_decomp=dc_decomp,
            mode=mode_from,
            perturbation_domain=perturbation_domain,
            model_input_dim=self.input_dim,
        )
        self.inputs_outputs_spec_to = InputsOutputsSpec(
            dc_decomp=dc_decomp,
            mode=mode_to,
            perturbation_domain=perturbation_domain,
            model_input_dim=self.input_dim,
        )

    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        compute_ibp_from_affine = self.mode_from == ForwardMode.AFFINE and self.mode_to != ForwardMode.AFFINE
        tight = self.mode_from == ForwardMode.HYBRID and self.mode_to != ForwardMode.AFFINE
        compute_dummy_affine = self.mode_from == ForwardMode.IBP and self.mode_to != ForwardMode.IBP
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec_from.get_fullinputs_from_inputsformode(
            inputs, compute_ibp_from_affine=compute_ibp_from_affine, tight=tight
        )

        if compute_dummy_affine:
            x = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            w = K.zeros_like(BatchedIdentityLike()(l_c))
            w_u = w
            b_u = u_c
            w_l = w
            b_l = l_c

        return self.inputs_outputs_spec_to.extract_outputsformode_from_fulloutputs(
            [x, u_c, w_u, b_u, l_c, w_l, b_l, h, g]
        )

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> List[Tuple[Optional[int], ...]]:
        (
            x_shape,
            u_c_shape,
            w_u_shape,
            b_u_shape,
            l_c_shape,
            w_l_shape,
            b_l_shape,
            h_shape,
            g_shape,
        ) = self.inputs_outputs_spec_from.get_fullinputshapes_from_inputshapesformode(input_shape)
        return self.inputs_outputs_spec_to.extract_inputshapesformode_from_fullinputshapes(
            [x_shape, u_c_shape, w_u_shape, b_u_shape, l_c_shape, w_l_shape, b_l_shape, h_shape, g_shape]
        )

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
