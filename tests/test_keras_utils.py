import keras
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Dense, Input
from numpy.testing import assert_almost_equal

from decomon.keras_utils import get_weight_index_from_name, share_layer_all_weights


def test_get_weight_index_from_name_nok_attribute():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    with pytest.raises(AttributeError):
        get_weight_index_from_name(layer=layer, weight_name="toto")


def test_get_weight_index_from_name_nok_index():
    layer = Dense(3, use_bias=False)
    layer(K.zeros((2, 1)))
    with pytest.raises(IndexError):
        get_weight_index_from_name(layer=layer, weight_name="bias")


def test_get_weight_index_from_name_ok():
    layer = Dense(3)
    layer(K.zeros((2, 1)))
    assert get_weight_index_from_name(layer=layer, weight_name="bias") in [0, 1]


def test_share_layer_all_weights_nok_original_layer_unbuilt():
    original_layer = Dense(3)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_nok_new_layer_built():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    new_layer(inp)
    with pytest.raises(ValueError):
        share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)


def test_share_layer_all_weights_ok():
    original_layer = Dense(3)
    inp = Input((1,))
    original_layer(inp)
    new_layer = original_layer.__class__.from_config(original_layer.get_config())
    share_layer_all_weights(original_layer=original_layer, new_layer=new_layer)

    # check same weights
    assert len(original_layer.weights) == len(new_layer.weights)
    for w in original_layer.weights:
        new_w = [ww for ww in new_layer.weights if ww.name == w.name][0]
        assert w is new_w
