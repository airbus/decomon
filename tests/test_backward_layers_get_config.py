import pytest
from tensorflow.keras.layers import Layer

from decomon.backward_layers.backward_layers import (
    BackwardActivation,
    BackwardBatchNormalization,
    BackwardConv2D,
    BackwardDense,
    BackwardDropout,
    BackwardFlatten,
    BackwardInputLayer,
    BackwardPermute,
    BackwardReshape,
)
from decomon.backward_layers.backward_maxpooling import BackwardMaxPooling2D
from decomon.backward_layers.backward_merge import (
    BackwardAdd,
    BackwardAverage,
    BackwardConcatenate,
    BackwardDot,
    BackwardMaximum,
    BackwardMinimum,
    BackwardMultiply,
    BackwardSubtract,
)
from decomon.backward_layers.crown import (
    Convert2BackwardMode,
    Convert2Mode,
    Fuse,
    MergeWithPrevious,
)
from decomon.backward_layers.deel_lip import BackwardGroupSort2
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import (
    DecomonActivation,
    DecomonBatchNormalization,
    DecomonConv2D,
    DecomonDense,
    DecomonDropout,
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
from decomon.layers.maxpooling import DecomonMaxPooling2D

try:
    from decomon.layers.deel_lip import DecomonGroupSort2
except ImportError:
    deel_lip_available = False
else:
    deel_lip_available = True


deel_lip_skip_reason = "deel-lip is not available"


def test_backward_layers():
    activation = "linear"
    sublayer = DecomonActivation(activation)
    layer = BackwardActivation(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    units = 2
    sublayer = DecomonDense(units=units, use_bias=False)
    layer = BackwardDense(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    filters = 2
    kernel_size = 3, 3
    sublayer = DecomonConv2D(filters=filters, kernel_size=kernel_size)
    layer = BackwardConv2D(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    rate = 0.9
    sublayer = DecomonDropout(rate=rate)
    layer = BackwardDropout(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    dims = (1, 2, 3)
    sublayer = DecomonPermute(dims=dims)
    layer = BackwardPermute(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    shape = (1, 2, 3)
    sublayer = DecomonReshape(target_shape=shape)
    layer = BackwardReshape(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    sublayer = DecomonBatchNormalization()
    layer = BackwardBatchNormalization(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__

    sublayer = Layer()
    for cls in [BackwardFlatten, BackwardInputLayer]:
        layer = cls(layer=sublayer)
        config = layer.get_config()
        assert config["layer"]["class_name"] == sublayer.__class__.__name__


backward2decomon = {
    BackwardAdd: DecomonAdd,
    BackwardAverage: DecomonAverage,
    BackwardConcatenate: DecomonConcatenate,
    # BackwardDot: DecomonDot,  # not yet implemented
    BackwardMaximum: DecomonMaximum,
    BackwardMinimum: DecomonMinimum,
    BackwardMultiply: DecomonMultiply,
    BackwardSubtract: DecomonSubtract,
}


def test_backward_merge():
    for backwardlayer_cls, decomonlayer_cls in backward2decomon.items():
        sublayer = decomonlayer_cls()
        layer = backwardlayer_cls(layer=sublayer)
        config = layer.get_config()
        assert config["layer"]["class_name"] == sublayer.__class__.__name__


@pytest.mark.skip("Not yet implemented.")
def test_backward_maxpooling():
    sublayer = DecomonMaxPooling2D()
    layer = BackwardMaxPooling2D(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__


@pytest.mark.skip("Not yet implemented.")
@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
def test_deel_lip():
    sublayer = DecomonGroupSort2()
    layer = BackwardGroupSort2(layer=sublayer)
    config = layer.get_config()
    assert config["layer"]["class_name"] == sublayer.__class__.__name__


def test_crown():
    mode = ForwardMode.AFFINE
    convex_domain = {}
    input_shape_layer = (1, 2, 4)
    backward_shape_layer = (2, 5, 10)

    layer = Fuse(mode=mode)
    config = layer.get_config()
    assert config["mode"] == mode

    layer = Convert2BackwardMode(mode=mode, convex_domain=convex_domain)
    config = layer.get_config()
    assert config["mode"] == mode
    assert "convex_domain" in config

    layer = MergeWithPrevious(input_shape_layer=input_shape_layer, backward_shape_layer=backward_shape_layer)
    config = layer.get_config()
    assert config["input_shape_layer"] == input_shape_layer
    assert config["backward_shape_layer"] == backward_shape_layer
