from decomon.layers.core import F_FORWARD
from decomon.layers.decomon_layers import (
    DecomonActivation,
    DecomonBatchNormalization,
    DecomonConv2D,
    DecomonDense,
    DecomonDropout,
    DecomonFlatten,
    DecomonInputLayer,
)
from decomon.layers.decomon_merge_layers import (
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonDot,
    DecomonMaximum,
    DecomonMinimum,
    DecomonMultiply,
    DecomonSubtract,
)
from decomon.layers.decomon_reshape import DecomonPermute, DecomonReshape
from decomon.layers.deel_lip import DecomonGroupSort, DecomonGroupSort2
from decomon.layers.maxpooling import DecomonMaxPooling2D


def test_decomon_reshape():
    dims = (1, 2, 3)
    mode = F_FORWARD.name
    layer = DecomonPermute(dims=dims, mode=mode)
    config = layer.get_config()
    assert config["dims"] == dims
    assert config["mode"] == mode

    shape = (1, 2, 3)
    layer = DecomonReshape(target_shape=shape, mode=mode)
    config = layer.get_config()
    assert config["target_shape"] == shape
    assert config["mode"] == mode


def test_deel_lip():
    layer = DecomonGroupSort()
    config = layer.get_config()

    layer = DecomonGroupSort2()
    config = layer.get_config()


def test_maxpooling():
    layer = DecomonMaxPooling2D()
    config = layer.get_config()
    assert "pool_size" in config


def test_decomon_merge_layers():
    for cls in [
        DecomonAdd,
        DecomonAverage,
        DecomonConcatenate,
        DecomonDot,
        DecomonMaximum,
        DecomonMinimum,
        DecomonMultiply,
        DecomonSubtract,
    ]:
        layer = cls()
        config = layer.get_config()
        assert "mode" in config
        if layer == DecomonConcatenate:
            assert "axis" in config
        elif layer == DecomonDot:
            assert "axes" in config


def test_decomon_layers():
    for activation in ["relu", "softmax"]:
        layer = DecomonActivation(activation=activation)
        config = layer.get_config()
        print(config)
        print(layer.activation)
        assert config["activation"] == activation

    units = 2
    layer = DecomonDense(units=units)
    config = layer.get_config()
    assert config["units"] == units

    filters = 2
    kernel_size = 3, 3
    layer = DecomonConv2D(filters=filters, kernel_size=kernel_size)
    config = layer.get_config()
    assert config["filters"] == filters
    assert config["kernel_size"] == kernel_size

    rate = 0.9
    layer = DecomonDropout(rate=rate)
    config = layer.get_config()
    assert config["rate"] == rate

    for cls in [DecomonBatchNormalization, DecomonInputLayer, DecomonFlatten]:
        layer = cls()
        config = layer.get_config()
        assert "mode" in config
        if layer == DecomonBatchNormalization:
            assert "axis" in config
        elif layer == DecomonInputLayer:
            assert "input_shape" in config
        elif layer == DecomonFlatten:
            assert "data_format" in config
