import keras
import keras.ops as K
import numpy as np
import pytest
import torch
from keras.layers import BatchNormalization

from decomon.layers.normalization.utils import BatchNormalization_kernel_constraint

from .conftest import build_keras_model, check_layer, empirical_check_layer, train_model


def _test_backward_batchnorm_empirical(input_shape, method, wo_linearity):
    keras_layer = BatchNormalization()
    input_dim = 30
    keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=True, output_dim=1)
    # train model
    # train_model(keras_model)

    weights = keras_layer.get_weights()

    gamma = weights[0]
    gamma[0] = -1
    gamma[-1] = -2
    weights[0] = gamma
    keras_layer.set_weights(weights)

    empirical_check_layer(
        keras_layer, input_shape, method=method, wo_linearity=wo_linearity, decimal=0, keras_model=keras_model
    )


# @pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid",  "crown", "crown-forward-ibp", "crown-forward-affine"])
@pytest.mark.parametrize(
    "method, data_format",
    [
        ("forward-affine", "channels_first"),
        ("forward-hybrid", "channels_first"),
        ("crown-forward-ibp", "channels_first"),
        ("crown", "channels_first"),
        ("forward-affine", "channels_last"),
        ("forward-hybrid", "channels_last"),
        ("crown-forward-ibp", "channels_last"),
        ("crown", "channels_last"),
    ],
)
def test_backward_batchnorm_empirical(method, data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 10)
    else:
        input_shape = (10, 2)
    _test_backward_batchnorm_empirical(input_shape, method, wo_linearity=False)
    _test_backward_batchnorm_empirical(input_shape, method, wo_linearity=True)

    if data_format == "channels_first":
        input_shape = (3, 11)
    else:
        input_shape = (11, 3)
    _test_backward_batchnorm_empirical(input_shape, method, wo_linearity=False)
    _test_backward_batchnorm_empirical(input_shape, method, wo_linearity=True)
