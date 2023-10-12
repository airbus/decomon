import keras_core as keras
import keras_core.ops as K
import pytest
from keras_core.layers import Dense

from decomon.keras_utils import get_weight_index_from_name


def test_get_weight_index_from_name_nok_attribute():
    layer = Dense(3, input_dim=1)
    layer(K.zeros((2, 1)))
    with pytest.raises(AttributeError):
        get_weight_index_from_name(layer=layer, weight_name="toto")


def test_get_weight_index_from_name_nok_index():
    layer = Dense(3, input_dim=1, use_bias=False)
    layer(K.zeros((2, 1)))
    with pytest.raises(IndexError):
        get_weight_index_from_name(layer=layer, weight_name="bias")


def test_get_weight_index_from_name_ok():
    layer = Dense(3, input_dim=1)
    layer(K.zeros((2, 1)))
    assert get_weight_index_from_name(layer=layer, weight_name="bias") in [0, 1]
