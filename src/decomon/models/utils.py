from typing import Any, Optional, Union

import keras
import keras.ops as K
import numpy as np
from keras import Model, Sequential
from keras import ops as K
from keras.layers import Activation, Input, Lambda, Layer
from keras.src import Functional
from keras.src.ops.node import Node

from decomon.core import (
    BallDomain,
    BoxDomain,
    ConvertMethod,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    Propagation,
    get_mode,
)
from decomon.keras_utils import (
    BatchedIdentityLike,
    is_a_merge_layer,
    share_weights_and_build,
)
from decomon.types import BackendTensor


def generate_perturbation_domain_input(
    model: Model,
    perturbation_domain: PerturbationDomain,
) -> keras.KerasTensor:
    model_input_shape = model.inputs[0].shape[1:]
    dtype = model.inputs[0].dtype

    input_shape_x = perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)
    return Input(shape=input_shape_x, dtype=dtype)


def prepare_inputs_for_layer(
    inputs: Union[tuple[keras.KerasTensor, ...], list[keras.KerasTensor], keras.KerasTensor]
) -> Union[tuple[keras.KerasTensor, ...], list[keras.KerasTensor], keras.KerasTensor]:
    """Prepare inputs for keras/decomon layers.

    Some Keras layers do not like list of tensors even with one single tensor.
    So we keep only the tensor in this case.

    """
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        if len(inputs) == 1:
            return inputs[0]
    return inputs


def wrap_outputs_from_layer_in_list(
    outputs: Union[tuple[keras.KerasTensor, ...], list[keras.KerasTensor], keras.KerasTensor]
) -> list[keras.KerasTensor]:
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    Args:
        x: target object to be normalized.

    Returns:
        A list.
    """
    if not isinstance(outputs, list):
        if isinstance(outputs, tuple):
            return list(outputs)
        else:
            return [outputs]
    else:
        return outputs


def split_activation(layer: Layer) -> list[Layer]:
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


def preprocess_layer(layer: Layer) -> list[Layer]:
    return split_activation(layer)


def is_input_node(node: Node) -> bool:
    return len(node.input_tensors) == 0


def get_output_nodes(model: Model) -> list[Node]:
    """Get list of output nodes ordered as model.outputs

    Args:
        model:

    Returns:

    """
    nodes_by_operation = {n.operation: n for subnodes in model._nodes_by_depth.values() for n in subnodes}
    return [nodes_by_operation[output._keras_history.operation] for output in model.outputs]


def get_depth_dict(model: Model) -> dict[int, list[Node]]:
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    nodes_list = []

    dico_depth: dict[int, int] = {}
    dico_nodes: dict[int, list[Node]] = {}

    def fill_dico(node: Node, dico_depth: Optional[dict[int, int]] = None) -> dict[int, int]:
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

    def call(self, inputs: list[BackendTensor], **kwargs: Any) -> list[BackendTensor]:
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

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> list[tuple[Optional[int], ...]]:
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

    def get_config(self) -> dict[str, Any]:
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


def preprocess_backward_bounds(
    backward_bounds: Optional[Union[keras.KerasTensor, list[keras.KerasTensor], list[list[keras.KerasTensor]]]],
    nb_model_outputs: int,
) -> Optional[list[list[keras.KerasTensor]]]:
    """Preprocess backward bounds to be used by `convert()`.

    Args:
        backward_bounds: backward bounds to propagate
        nb_model_outputs: number of outputs of the keras model to convert

    Returns:
        formatted backward bounds

    Backward bounds can be given as
    - None or empty list => backward bounds to propagate will be identity
    - a single list (potentially partially filled) => same backward bounds on all model outputs (assuming that all outputs have same shape)
    - a list of list : different backward bounds for each model output

    Which leads to the following formatting:
    - None, [], or [[]] -> None
    - single keras tensor w -> [[w, 0, w, 0]] * nb_model_outputs
    - [w] -> idem
    - [w, b] -> [[w, b, w, b]] * nb_model_outputs
    - [w_l, b_l, w_u, b_u] -> [[w_l, b_l, w_u, b_u]] * nb_model_outputs
    - [[w_l, b_l, w_u, b_u]] -> [[w_l, b_l, w_u, b_u]] * nb_model_outputs
    - list of lists of tensors [w_l[i], b_l[i], w_u[i], b_u[i]]_i -> we enforce each sublist to have 4 elements

    """
    if backward_bounds is None:
        # None
        return None
    if isinstance(backward_bounds, keras.KerasTensor):
        # single tensor w
        w = backward_bounds
        b = K.zeros_like(w[:, 0])
        backward_bounds = [w, b, w, b]
    if len(backward_bounds) == 0:
        return None
    else:
        if isinstance(backward_bounds[0], keras.KerasTensor):
            # list of tensors
            if len(backward_bounds) == 1:
                # single tensor w
                return preprocess_backward_bounds(backward_bounds=backward_bounds[0], nb_model_outputs=nb_model_outputs)
            elif len(backward_bounds) == 2:
                # w, b
                w, b = backward_bounds
                return preprocess_backward_bounds(backward_bounds=[w, b, w, b], nb_model_outputs=nb_model_outputs)
            elif len(backward_bounds) == 4:
                return [backward_bounds] * nb_model_outputs
            else:
                raise ValueError(
                    "If backward_bounds is given as a list of tensors, it should have 1, 2, or 4 elements."
                )
        else:
            # list of list of tensors
            if len(backward_bounds) == 1:
                return [backward_bounds[0]] * nb_model_outputs
            elif len(backward_bounds) != nb_model_outputs:
                raise ValueError(
                    "If backward_bounds is given as a list of tensors, it should have nb_model_ouptputs elements."
                )
            elif not all([len(backward_bounds_i) == 4 for backward_bounds_i in backward_bounds]):
                raise ValueError(
                    "If backward_bounds is given as a list of tensors, each sublist should have 4 elements (w_l_, b_l, w_u, b_u)."
                )
            else:
                return backward_bounds


def get_ibp_affine_from_method(method: ConvertMethod) -> tuple[bool, bool]:
    method = ConvertMethod(method)
    if method in [ConvertMethod.FORWARD_IBP, ConvertMethod.CROWN_FORWARD_IBP]:
        return True, False
    if method in [ConvertMethod.FORWARD_AFFINE, ConvertMethod.CROWN_FORWARD_AFFINE]:
        return False, True
    if method in [ConvertMethod.FORWARD_HYBRID, ConvertMethod.CROWN_FORWARD_HYBRID]:
        return True, True
    if method == ConvertMethod.CROWN:
        return False, True
    return True, True


def get_final_ibp_affine_from_method(method: ConvertMethod) -> tuple[bool, bool]:
    method = ConvertMethod(method)
    if method in [ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_HYBRID]:
        final_ibp = True
    else:
        final_ibp = False
    if method == ConvertMethod.FORWARD_IBP:
        final_affine = False
    else:
        final_affine = True

    return final_ibp, final_affine


def method2propagation(method: ConvertMethod) -> list[Propagation]:
    if method == ConvertMethod.CROWN:
        return [Propagation.BACKWARD]
    elif method in [ConvertMethod.FORWARD_IBP, ConvertMethod.FORWARD_AFFINE, ConvertMethod.FORWARD_HYBRID]:
        return [Propagation.FORWARD]
    else:
        return [Propagation.FORWARD, Propagation.BACKWARD]
