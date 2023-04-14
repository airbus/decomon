import numpy as np
import pytest
import tensorflow.keras.backend as K
from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model, Sequential

from decomon.models.convert import split_activations_in_keras_model


def test_split_activations_in_keras_model_no_inputshape_ko():
    layers = [
        Conv2D(
            10,
            kernel_size=(3, 3),
            activation="relu",
            data_format="channels_last",
        ),
        Flatten(),
        Dense(1),
    ]
    model = Sequential(layers)
    with pytest.raises(ValueError):
        converted_model = split_activations_in_keras_model(model)


def test_split_activations_in_keras_model(toy_model):
    converted_model = split_activations_in_keras_model(toy_model)
    assert isinstance(converted_model, Model)
    # check no more activation functions in non-activation layers
    for layer in converted_model.layers:
        activation = layer.get_config().get("activation", None)
        assert isinstance(layer, Activation) or activation is None or activation == "linear"
    # check same outputs
    input_shape_wo_batchsize = toy_model.input_shape[1:]
    input_shape_with_batch_size = (5,) + input_shape_wo_batchsize
    flatten_dim = np.prod(input_shape_with_batch_size)
    inputs_np = np.linspace(-1, 1, flatten_dim).reshape(input_shape_with_batch_size)
    output_np_ref = toy_model(inputs_np).numpy()
    output_np_new = converted_model(inputs_np).numpy()
    assert_almost_equal(output_np_new, output_np_ref)
