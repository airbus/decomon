import pytest
from keras.layers import Add, Conv2D, Dense, Input, Layer, Reshape

from decomon.layers.convert import to_decomon
from decomon.layers.utils import is_a_merge_layer


class MyLayer(Layer):
    """Mock layer unknown from decomon."""

    ...


class MyMerge(Layer):
    """Mock merge layer unknown from decomon."""

    def _merge_function(self, inputs):
        return inputs


def test_is_merge_layer():
    layer = MyMerge()
    assert is_a_merge_layer(layer)
    layer = MyLayer()
    assert not is_a_merge_layer(layer)


def test_to_decomon_merge_not_built_ko():
    layer = MyMerge()
    with pytest.raises(ValueError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_not_built_ko():
    layer = MyLayer()
    with pytest.raises(ValueError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_merge_not_implemented_ko():
    layer = MyMerge()
    layer.built = True
    with pytest.raises(NotImplementedError):
        to_decomon(layer, input_dim=1)


def test_to_decomon_not_implemented_ko():
    layer = MyLayer()
    layer.built = True
    with pytest.raises(NotImplementedError):
        to_decomon(layer, input_dim=1)


@pytest.mark.parametrize(
    "layer_class, layer_kwargs, input_shape_wo_batchsize, nb_inputs",
    [
        (Dense, {"units": 3, "use_bias": True}, (1,), 1),
        (Conv2D, {"filters": 2, "kernel_size": (3, 3), "use_bias": True}, (64, 64, 3), 1),
        (Dense, {"units": 3, "use_bias": False}, (1,), 1),
        (Conv2D, {"filters": 2, "kernel_size": (3, 3), "use_bias": False}, (64, 64, 3), 1),
        (Reshape, {"target_shape": (72,)}, (6, 6, 2), 1),
        (Add, {}, (1,), 2),
    ],
)
def test_to_decomon_ok(layer_class, layer_kwargs, input_shape_wo_batchsize, nb_inputs):
    layer = layer_class(**layer_kwargs)
    # init input_shape and weights
    # input_tensors + build layer
    if nb_inputs == 1:
        input_tensor = Input(input_shape_wo_batchsize)
        layer(input_tensor)
    else:
        input_tensors = [Input(input_shape_wo_batchsize) for _ in range(nb_inputs)]
        layer(input_tensors)
    decomon_layer = to_decomon(layer, input_dim=1)
    # check trainable weights
    for i in range(len(layer.weights)):
        assert layer.weights[i] is decomon_layer.weights[i]
