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
