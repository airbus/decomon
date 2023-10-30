from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras.layers import Activation, Input, Layer

import decomon.layers.decomon_layers
import decomon.layers.decomon_merge_layers
import decomon.layers.decomon_reshape
import decomon.layers.deel_lip
import decomon.layers.maxpooling
from decomon.core import BoxDomain, ForwardMode, PerturbationDomain, Slope, get_mode
from decomon.layers.core import DecomonLayer

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
    perturbation_domain: Optional[PerturbationDomain] = None,
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
            of the input perturbation domain
        slope:
        dc_decomp: boolean that indicates whether we return a difference
            of convex decomposition of our layer
        perturbation_domain: the type of perturbation domain
        ibp: boolean that indicates whether we propagate constant bounds
        affine: boolean that indicates whether we propagate affine
            bounds

    Returns:
        the associated DecomonLayer
    """

    # get class name
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()

    mode = get_mode(ibp=ibp, affine=affine)
    layer_decomon = _to_decomon_wo_input_init(
        layer=layer,
        namespace=_mapping_name2class,
        slope=slope,
        dc_decomp=dc_decomp,
        perturbation_domain=perturbation_domain,
        finetune=finetune,
        mode=mode,
        shared=shared,
        fast=fast,
    )

    input_tensors = _prepare_input_tensors(
        layer=layer, input_dim=input_dim, dc_decomp=dc_decomp, perturbation_domain=perturbation_domain, mode=mode
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
    perturbation_domain: Optional[PerturbationDomain] = None,
    finetune: bool = False,
    mode: ForwardMode = ForwardMode.HYBRID,
    shared: bool = True,
    fast: bool = True,
) -> DecomonLayer:
    if perturbation_domain is None:
        perturbation_domain = BoxDomain()
    class_name = layer.__class__.__name__
    # check if layer has a built argument that built is set to True
    if hasattr(layer, "built"):
        if not layer.built:
            raise ValueError(f"the layer {layer.name} has not been built yet")
    decomon_class_name = f"Decomon{class_name}"
    config_layer = layer.get_config()
    config_layer["name"] = layer.name + "_decomon"
    config_layer["dc_decomp"] = dc_decomp
    config_layer["perturbation_domain"] = perturbation_domain

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
    layer: Layer, input_dim: int, dc_decomp: bool, perturbation_domain: PerturbationDomain, mode: ForwardMode
) -> List[keras.KerasTensor]:
    original_input_shapes = get_layer_input_shape(layer)
    decomon_input_shapes: List[List[Optional[int]]] = [list(input_shape[1:]) for input_shape in original_input_shapes]
    n_input = len(decomon_input_shapes)
    x_input_shape = perturbation_domain.get_x_input_shape(input_dim)
    x_input = Input(x_input_shape, dtype=layer.dtype)
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


SingleInputShapeType = Tuple[Optional[int], ...]


def get_layer_input_shape(layer: Layer) -> List[SingleInputShapeType]:
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Args:
        layer:

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """

    if not layer._inbound_nodes:
        raise AttributeError(f'The layer "{layer.name}" has never been called ' "and thus has no defined input shape.")
    all_input_shapes = set([str([tensor.shape for tensor in node.input_tensors]) for node in layer._inbound_nodes])
    if len(all_input_shapes) == 1:
        return [tensor.shape for tensor in layer._inbound_nodes[0].input_tensors]
    else:
        raise AttributeError(
            'The layer "' + str(layer.name) + '" has multiple inbound nodes, '
            "with different input shapes. Hence "
            'the notion of "input shape" is '
            "ill-defined for the layer. "
        )
