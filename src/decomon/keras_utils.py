from typing import List

import keras
from keras.layers import Layer


def get_weight_index(layer: Layer, weight: keras.Variable) -> int:
    """Get weight index among layer tracked weights

    Args:
        layer: layer we are looking
        weight: weight supposed to be part of tracked weights by the layer

    Returns:
        the index of the weight in `layer.weights` list

    Raises:
        IndexError: if `weight` is not part of `layer.weights`

    """
    indexes = [i for i, w in enumerate(layer.weights) if w is weight]
    try:
        return indexes[0]
    except IndexError:
        raise IndexError(f"The weight {weight} is not tracked by the layer {layer}.")


def get_weight_index_from_name(layer: Layer, weight_name: str) -> int:
    """Get weight index among layer tracked weights

    Args:
        layer: layer we are looking
        weight_name: name of the weight supposed to be part of tracked weights by the layer

    Returns:
        the index of the weight in `layer.weights` list

    Raises:
        AttributeError: if `weight_name` is not the name of an attribute of `layer`
        IndexError: if the corresponding layer attribute is not part of `layer.weights`

    """
    weight = getattr(layer, weight_name)
    try:
        return get_weight_index(layer=layer, weight=weight)
    except IndexError:
        raise IndexError(f"The weight {weight_name} is not tracked by the layer {layer}.")


def reset_layer(new_layer: Layer, original_layer: Layer, weight_names: List[str]) -> None:
    """Reset some weights of a layer by using the weights of another layer.

    Args:
        new_layer: the decomon layer whose weights will be updated
        original_layer: the layer used to update the weights
        weight_names: the names of the weights to update

    Returns:

    """
    if not original_layer.built:
        raise ValueError(f"the layer {original_layer.name} has not been built yet")
    if not new_layer.built:
        raise ValueError(f"the layer {new_layer.name} has not been built yet")
    else:
        new_params = new_layer.get_weights()
        original_params = original_layer.get_weights()
        for weight_name in weight_names:
            new_params[get_weight_index_from_name(new_layer, weight_name)] = original_params[
                get_weight_index_from_name(original_layer, weight_name)
            ]
        new_layer.set_weights(new_params)


def reset_layer_all_weights(new_layer: Layer, original_layer: Layer) -> None:
    """Reset all the weights of a layer by using the weights of another layer.

    Args:
        new_layer: the decomon layer whose weights will be updated
        original_layer: the layer used to update the weights

    Returns:

    """
    reset_layer(new_layer=new_layer, original_layer=original_layer, weight_names=[w.name for w in new_layer.weights])
