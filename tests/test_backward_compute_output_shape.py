import pytest
from keras.layers import Activation, InputLayer, Reshape

from decomon.backward_layers.convert import to_backward
from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_layers import (
    DecomonActivation,
    DecomonBatchNormalization,
    DecomonConv2D,
    DecomonDense,
    DecomonFlatten,
)
from decomon.layers.decomon_merge_layers import DecomonAdd, DecomonConcatenate
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.layers.maxpooling import DecomonMaxPooling2D

try:
    import deel.lip
except ImportError:
    deel_lip_available = False
    DecomonGroupSort2 = "DecomonGroupSort2"
else:
    deel_lip_available = True
    from decomon.layers.deel_lip import DecomonGroupSort2


@pytest.mark.parametrize(
    "layer_class, layer_kwargs",
    [
        (DecomonDense, dict(units=3, use_bias=True)),
        (DecomonFlatten, dict()),
        (DecomonBatchNormalization, dict(center=True, scale=True)),
        (DecomonBatchNormalization, dict(center=False, scale=False)),
        (DecomonActivation, dict(activation="linear")),
        (DecomonActivation, dict(activation="relu")),
        (Activation, dict(activation="linear")),
        (Activation, dict(activation="relu")),
        (DecomonConv2D, dict(filters=10, kernel_size=(3, 3))),
        (DecomonReshape, dict(target_shape=(1, -1, 1))),
        (Reshape, dict(target_shape=(1, -1, 1))),
        (DecomonGroupSort2, dict()),
    ],
)
@pytest.mark.parametrize("dc_decomp", [False])  # limit dc_decomp
@pytest.mark.parametrize("n", [0])  # limit 1d cases
@pytest.mark.parametrize("odd", [0])  # limit multid cases
@pytest.mark.parametrize("data_format", ["channels_last"])  # limit images cases
def test_compute_output_shape(
    helpers,
    mode,
    dc_decomp,
    layer_class,
    layer_kwargs,
    inputs_for_mode,  # decomon inputs: symbolic tensors
    input_ref,  # keras input: symbolic tensor
    inputs_for_mode_,  # decomon inputs: numpy arrays
    inputs_metadata,  # inputs metadata: data_format, ...
):
    # skip nonsensical combinations
    if (layer_class == DecomonBatchNormalization or layer_class == DecomonMaxPooling2D) and dc_decomp:
        pytest.skip(f"{layer_class} with dc_decomp=True not yet implemented.")
    if (layer_class == DecomonConv2D or layer_class == DecomonMaxPooling2D) and len(input_ref.shape) < 4:
        pytest.skip(f"{layer_class} applies only on image-like inputs.")
    if layer_class == DecomonGroupSort2:
        if not deel_lip_available:
            pytest.skip("deel-lip is not available")

    # add data_format for convolution and maxpooling
    if layer_class in (DecomonConv2D, DecomonMaxPooling2D):
        layer_kwargs["data_format"] = inputs_metadata["data_format"]

    # construct and build original layer (decomon or keras)
    if issubclass(layer_class, DecomonLayer):
        layer = layer_class(mode=mode, dc_decomp=dc_decomp, **layer_kwargs)
        layer(inputs_for_mode)
    else:  # keras layer
        layer = layer_class(**layer_kwargs)
        layer(input_ref)

    # get backward layer
    backward_layer = to_backward(layer, mode=mode)

    # check symbolic tensor output shapes
    inputshapes = [i.shape for i in inputs_for_mode]
    outputshapes = backward_layer.compute_output_shape(inputshapes)
    outputs = backward_layer(inputs_for_mode)
    assert [o.shape for o in outputs] == outputshapes

    # check output shapes for concrete call
    outputs_ = backward_layer(inputs_for_mode_)
    # compare without batch sizes
    assert [tuple(o.shape)[1:] for o in outputs_] == [s[1:] for s in outputshapes]
