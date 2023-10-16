import pytest
from keras_core.layers import Dense, Input

from decomon.layers.convert import get_layer_input_shape


def test_get_layer_input_shape_nok_uncalled():
    layer = Dense(3)
    with pytest.raises(AttributeError):
        get_layer_input_shape(layer)


def test_get_layer_input_shape_nok_multiple_shapes():
    layer = Dense(3)
    layer(Input((1,)))
    layer(Input((1, 1)))
    with pytest.raises(AttributeError):
        get_layer_input_shape(layer)


def test_get_layer_input_shape_ok():
    layer = Dense(3)
    layer(Input((1,)))
    input_shape = get_layer_input_shape(layer)
    assert input_shape == [(None, 1)]
