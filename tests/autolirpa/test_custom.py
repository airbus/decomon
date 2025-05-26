# custom layer for max

import keras
import pytest
import torch
import torch.nn as nn
from keras_custom.layers.reduce.base_reduce import BaseAxisKeepdimsLayer

from decomon.layers.custom import DecomonMax

from .conftest import check_layer, check_layer_linear


class Max(BaseAxisKeepdimsLayer):
    """
    Custom Keras Layer that computes the maximum value along a specified axis of the input tensor.
    Inherits axis and keepdims attributes from BaseAxisKeepdimsLayer.
    """

    def call(self, inputs_):
        """Computes the maximum value along the specified axis, retaining dimensions if keepdims is True."""
        return keras.ops.max(inputs_, axis=self.axis, keepdims=self.keepdims)


class TorchMax(nn.Module):
    def __init__(self, axis) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.max(x, axis=self.axis).values


def test_max(method="crown"):
    pytest.skip()
    input_shape = (10, 10)
    axis = -1
    keras_layer = Max(axis=axis)
    torch_layer = TorchMax(axis=axis)
    mapping_keras2decomon_classes = {Max: DecomonMax}
    check_layer(
        keras_layer,
        torch_layer,
        input_shape,
        method=method,
        decimal=5,
        mapping_keras2decomon_classes=mapping_keras2decomon_classes,
    )
