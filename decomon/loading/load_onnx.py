# load a keras network from an onnx file

# Python code for automatic conversion in channel last from an onnx file

import numpy as np

# Requirements
import onnx
from onnx2keras import onnx_to_keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.generic_utils import to_list

CHANNEL_FIRST = "channels_first"
CHANNEL_LAST = "channels_last"


def get_all_input_names(onnx_model):
    inputs_ = onnx_model.graph.input
    return [input_.name for input_ in inputs_]


# check for issue with data_format
def check_compatibility_data_format(k_model, allowed_format=None):
    if allowed_format is None:
        allowed_format = ["channels_last"]
    for layer in k_model.layers:
        if hasattr(layer, "data_format") and layer.data_format not in allowed_format:
            return True

    return False


def get_data_format(node):
    layer_ = node.outbound_layer

    if hasattr(layer_, "data_format"):
        return layer_.data_format
    if getattr(layer_, "get_weights") and len(layer_.get_weights()):
        return CHANNEL_LAST
    return "None"


def get_tensor_input(node):
    parents = node.parent_nodes
    if len(parents) > 1:
        raise NotImplementedError()

    parent = parents[0]
    if len(parent.outbound_layer.get_input_shape_at(0)[1:]) == 3:
        return parent.outbound_layer.get_input_shape_at(0)[1:]
    else:
        return get_tensor_input(parent)


def clone(layer, data_format=CHANNEL_LAST, prev_format=CHANNEL_LAST, node=None):
    layer_ = layer
    if data_format == CHANNEL_FIRST:
        # change config
        class_name = layer.__class__.__name__
        if hasattr(layer, "built"):
            if not layer.built:
                raise ValueError(f"the layer {layer.name} has not been built yet")

        config_layer = layer.get_config()
        config_layer["name"] = layer.name + "_last"
        config_layer["data_format"] = CHANNEL_LAST

        layer_ = globals()[class_name].from_config(config_layer)

        input_shape = list(layer.get_input_shape_at(0)[1:])
        input_shape_ = [1] + input_shape[1:] + [input_shape[0]]
        _ = layer_(np.zeros(input_shape_))
        layer_.set_weights(layer.get_weights())

    else:
        if isinstance(layer, Dense):
            if CHANNEL_FIRST in prev_format:
                # do some pre-processing
                W, b = layer.get_weights()
                # retrive input_shape before Flatten
                input_shape = get_tensor_input(node)
                transpose_indices = np.array(get_input_transpose(input_shape), dtype="int")
                W = W[transpose_indices, :]
                layer_.set_weights([W, b])

            # retrive input_shape to guess the permutation...
        elif isinstance(layer, InputLayer):
            # do permutation
            input_shape = list(layer.input_shape[0][1:])
            layer_ = InputLayer(input_shape)
        else:

            if not isinstance(layer, Lambda):
                class_name = layer.__class__.__name__
                config_layer = layer.get_config()
                config_layer["name"] = layer.name + "_last"
                layer_ = globals()[class_name].from_config(config_layer)
                input_shape = list(layer.get_input_shape_at(0)[1:])
                if len(input_shape) > 2:
                    input_shape_ = [1] + input_shape[1:] + [input_shape[0]]
                    _ = layer_(np.zeros(input_shape_))

    return layer_


def update_dico(node, dico_nodes, layer_names):
    if id(node) in dico_nodes:
        return None

    layer = node.outbound_layer
    if isinstance(layer, InputLayer):

        input_shape_ = list(layer.get_input_shape_at(0)[1:])

        var_ = Input(input_shape_[1:] + [input_shape_[0]])
        dico_nodes[id(node)] = to_list(var_)
    else:

        p_format = get_parents_format_(node)

        layer_format = get_data_format(node)
        print(p_format, layer_format, node.outbound_layer)
        layer_ = clone(node.outbound_layer, layer_format, p_format, node)
        parents = node.parent_nodes
        inputs_ = []
        for p_node in parents:
            try:
                inputs_ += dico_nodes[id(p_node)]
            except KeyError:
                update_dico(p_node, dico_nodes, layer_names)
                inputs_ += dico_nodes[id(p_node)]

        if len(inputs_) == 1:
            output = layer_(inputs_[0])
        else:
            output = layer_(inputs_)

        dico_nodes[id(node)] = to_list(output)

    return None


def clone_first_2_last(model):
    dico_nodes = {}
    layer_names = [l.name for l in model.layers]

    if not check_compatibility_data_format(model):
        return model

    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.reverse()

    input_models = []

    # because of an inconsistency of depth with onnx
    # it is better to build the information for the input first

    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:

            update_dico(node, dico_nodes, layer_names)
            if isinstance(node.outbound_layer, InputLayer):
                input_models += dico_nodes[id(node)]

    # find input models

    outputs = []
    output_nodes = model._nodes_by_depth[0]
    for node in output_nodes:
        outputs += dico_nodes[id(node)]
    # find output
    return Model(input_models, outputs)


def get_input_transpose(input_shape):
    n_c, n_j, n_k = input_shape

    count = 0
    toto = np.zeros((n_c, n_j, n_k))
    for i in range(n_c):
        for j in range(n_j):
            for k in range(n_k):
                toto[i, j, k] = count
                count += 1

    # permute dimensions
    return np.transpose(toto, (1, 2, 0)).flatten()


def get_parents_format_(node_):
    # recursive approach until encountering a layer that has a data_format or has some trainable weights
    if isinstance(node_, list):
        import pdb

        pdb.set_trace()
    parents = to_list(node_.parent_nodes)
    parents_format = []

    for node_p in parents:
        layer_ = node_p.outbound_layer
        if hasattr(layer_, "data_format"):
            parents_format.append(layer_.data_format)
        else:
            if getattr(layer_, "get_weights") and len(layer_.get_weights()):
                parents_format.append(CHANNEL_LAST)
            else:
                parent_rec = to_list(node_p.parent_nodes)
                if len(parent_rec) == 0:
                    parents_format.append("None")
                for p_rec in parent_rec:
                    toto = get_data_format(p_rec)
                    if toto == "None":
                        toto = get_parents_format_(p_rec)
                    if not len(toto):
                        parents_format.append(get_data_format(p_rec))
                    else:
                        if isinstance(toto, list):
                            toto = toto[0]
                        parents_format.append(toto)
    return parents_format


def load_onnx_2_keras(onnx_filename):
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    k_model = onnx_to_keras(onnx_model, get_all_input_names(onnx_model))
    return clone_first_2_last(k_model)
