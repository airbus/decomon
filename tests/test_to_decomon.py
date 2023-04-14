import pytest
from tensorflow.keras.layers import Layer

from decomon.layers.decomon_layers import to_decomon

try:
    from keras.layers.merge import _Merge as Merge
except ModuleNotFoundError:
    from tensorflow.python.keras.layers.merge import _Merge as Merge


def test_to_decomon_merge_not_built_ko():
    class MyMerge(Merge):
        ...

    layer = MyMerge()
    with pytest.raises(ValueError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_not_built_ko():
    class MyLayer(Layer):
        ...

    layer = MyLayer()
    with pytest.raises(ValueError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_merge_not_implemented_ko():
    class MyMerge(Merge):
        ...

    layer = MyMerge()
    layer.built = True
    with pytest.raises(NotImplementedError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_not_implemented_ko():
    class MyLayer(Layer):
        ...

    layer = MyLayer()
    layer.built = True
    with pytest.raises(NotImplementedError):
        to_decomon(layer, input_dim=1)
