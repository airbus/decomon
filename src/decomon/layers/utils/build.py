from typing import List

import numpy as np  # type:ignore
from keras.layers import Layer  # type:ignore
from keras.models import Sequential  # type:ignore


def pre_built(layer: Layer, input_shape_wo_batch: List[int]):
    """
    This function ensures that the provided Keras layer is built by passing a dummy input through it.
    If the layer has not been built yet, it creates a toy model with the given layer,
    generates a dummy input based on the provided shape, and then performs a forward pass
    to trigger the building of the layer.

    Args:
        layer: A Keras layer that needs to be built.
        input_shape_wo_batch: A tuple representing the input shape excluding the batch size.

    Returns:
        None: This function does not return anything.
    """
    if not layer.built:
        toy_model = Sequential([layer])
        input = np.zeros([1] + input_shape_wo_batch)
        _ = toy_model(input)
