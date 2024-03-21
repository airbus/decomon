from typing import Optional, Union

import keras
import numpy as np
from keras import Model, Sequential
from keras.layers import Activation, Input, Layer
from keras.src import Functional
from keras.src.ops.node import Node

from decomon.constants import ConvertMethod, Propagation
from decomon.keras_utils import share_weights_and_build
from decomon.perturbation_domain import PerturbationDomain


def generate_perturbation_domain_input(
    model: Model, perturbation_domain: PerturbationDomain, name: str = "perturbation_domain_input"
) -> keras.KerasTensor:
    model_input_shape = model.inputs[0].shape[1:]
    dtype = model.inputs[0].dtype

    input_shape_x = perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)
    return Input(shape=input_shape_x, dtype=dtype, name=name)


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


def remove_last_softmax_layers(model: Model) -> Model:
    """Remove for each output the last layer if it is a softmax activation

    NB: this should be applied after the split of activations, so that we only need
    to check for Activation layers.

    Args:
        model: original keras model

    Returns:
        model without the last softmax layers

    It will return the same model if no such softmax have to be removed, else construct a new functional model.

    """
    output_nodes = get_output_nodes(model)
    new_outputs = []
    has_changed = False
    for output_node in output_nodes:
        layer = output_node.operation
        # NB: activations have been already split => we need only to check for Activation layers
        if isinstance(layer, Activation) and layer.get_config()["activation"] == "softmax":
            # softmax: take parent nodes outputs instead
            for parent in output_node.parent_nodes:
                new_outputs += parent.outputs
            has_changed = True
        else:
            # no softmax: keep same outputs
            new_outputs += output_node.outputs
    if has_changed:
        return Model(model.inputs, new_outputs)
    else:
        return model


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


def ensure_functional_model(model: Model) -> Functional:
    if isinstance(model, Functional):
        return model
    elif isinstance(model, Sequential):
        model = Model(model.inputs, model.outputs)
        assert isinstance(model, Functional)  # should be the case after passage in Model.__init__()
        return model
    else:
        raise NotImplementedError("Decomon model available only for functional or sequential models.")


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
