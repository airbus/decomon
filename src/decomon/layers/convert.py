from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Layer

import decomon.layers.decomon_layers
import decomon.layers.decomon_merge_layers
import decomon.layers.decomon_reshape
import decomon.layers.deel_lip
import decomon.layers.maxpooling
from decomon.layers.core import DEEL_LIP, DecomonLayer, ForwardMode, get_mode
from decomon.utils import ConvexDomainType, Slope

# mapping between decomon class names and actual classes
_mapping_name2class = vars(decomon.layers.decomon_layers)
_mapping_name2class.update(vars(decomon.layers.decomon_merge_layers))
_mapping_name2class.update(vars(decomon.layers.decomon_reshape))
_mapping_name2class.update(vars(decomon.layers.deel_lip))
_mapping_name2class.update(vars(decomon.layers.maxpooling))


def to_decomon(
    layer: Layer,
    input_dim: int,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    ibp: bool = True,
    affine: bool = True,
    shared: bool = True,
    fast: bool = True,
) -> DecomonLayer:
    """Transform a standard keras layer into a Decomon layer.

    Type of layer is tested to know how to transform it into a DecomonLayer of the good type.
    If type is not treated yet, raises an TypeError

    Args:
        layer: a Keras Layer
        input_dim: an integer that represents the dim
            of the input convex domain
        slope:
        dc_decomp: boolean that indicates whether we return a difference
            of convex decomposition of our layer
        convex_domain: the type of convex domain
        ibp: boolean that indicates whether we propagate constant bounds
        affine: boolean that indicates whether we propagate affine
            bounds

    Returns:
        the associated DecomonLayer
    """

    # get class name
    if convex_domain is None:
        convex_domain = {}

    mode = get_mode(ibp=ibp, affine=affine)
    original_class_name = layer.__class__.__name__
    layer_decomon: Optional[DecomonLayer] = None
    for k in range(2):  # two runs before sending a failure
        try:
            layer_decomon = _to_decomon_wo_input_init(
                layer=layer,
                namespace=_mapping_name2class,
                slope=slope,
                dc_decomp=dc_decomp,
                convex_domain=convex_domain,
                finetune=finetune,
                mode=mode,
                shared=shared,
                fast=fast,
            )

            break
        except NotImplementedError:
            if hasattr(layer, "vanilla_export"):
                shared = False  # checking with Deel-LIP
                layer_ = layer.vanilla_export()
                layer_(layer.input)
                layer = layer_

    if layer_decomon is None:
        raise NotImplementedError(f"The decomon version of {original_class_name} is not yet implemented.")

    input_tensors = _prepare_input_tensors(
        layer=layer, input_dim=input_dim, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode
    )

    layer_decomon(input_tensors)
    layer_decomon.reset_layer(layer)

    # return layer_decomon
    return layer_decomon


def _to_decomon_wo_input_init(
    layer: Layer,
    namespace: Dict[str, Any],
    slope: Union[str, Slope] = Slope.V_SLOPE,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    mode: ForwardMode = ForwardMode.HYBRID,
    shared: bool = True,
    fast: bool = True,
) -> DecomonLayer:
    if convex_domain is None:
        convex_domain = {}
    class_name = layer.__class__.__name__
    # remove deel-lip dependency
    if class_name[: len(DEEL_LIP)] == DEEL_LIP:
        class_name = class_name[len(DEEL_LIP) :]
    # check if layer has a built argument that built is set to True
    if hasattr(layer, "built"):
        if not layer.built:
            raise ValueError(f"the layer {layer.name} has not been built yet")
    decomon_class_name = f"Decomon{class_name}"
    config_layer = layer.get_config()
    config_layer["name"] = layer.name + "_decomon"
    config_layer["dc_decomp"] = dc_decomp
    config_layer["convex_domain"] = convex_domain

    config_layer["mode"] = mode
    config_layer["finetune"] = finetune
    config_layer["slope"] = slope
    config_layer["shared"] = shared
    config_layer["fast"] = fast
    if not isinstance(layer, Activation):
        config_layer.pop("activation", None)  # Hyp: no non-linear activation in dense or conv2d layers

    try:
        layer_decomon = namespace[decomon_class_name].from_config(config_layer)
    except:
        raise NotImplementedError(f"The decomon version of {class_name} is not yet implemented.")

    layer_decomon.share_weights(layer)
    return layer_decomon


def _prepare_input_tensors(
    layer: Layer, input_dim: int, dc_decomp: bool, convex_domain: Dict[str, Any], mode: ForwardMode
) -> List[tf.Tensor]:
    original_input_shapes = layer.input_shape
    if not isinstance(original_input_shapes, list):
        original_input_shapes = [original_input_shapes]

    decomon_input_shapes = [list(input_shape[1:]) for input_shape in original_input_shapes]
    n_input = len(decomon_input_shapes)

    if len(convex_domain) == 0 or convex_domain["name"] == ConvexDomainType.BOX:
        x_input = Input((2, input_dim), dtype=layer.dtype)
    else:
        x_input = Input((input_dim,), dtype=layer.dtype)

    w_input = [Input(tuple([input_dim] + decomon_input_shapes[i])) for i in range(n_input)]
    y_input = [Input(tuple(decomon_input_shapes[i])) for i in range(n_input)]

    if mode == ForwardMode.HYBRID:
        nested_input_list = [
            [x_input, y_input[i], w_input[i], y_input[i], y_input[i], w_input[i], y_input[i]] for i in range(n_input)
        ]
    elif mode == ForwardMode.IBP:
        nested_input_list = [[y_input[i], y_input[i]] for i in range(n_input)]
    elif mode == ForwardMode.AFFINE:
        nested_input_list = [[x_input, w_input[i], y_input[i], w_input[i], y_input[i]] for i in range(n_input)]
    else:
        raise ValueError(f"Unknown mode {mode}")

    flatten_input_list = [tensor for input_list in nested_input_list for tensor in input_list]

    if dc_decomp:
        if n_input == 1:
            flatten_input_list += [y_input[0], y_input[0]]
        else:
            raise NotImplementedError()

    return flatten_input_list
