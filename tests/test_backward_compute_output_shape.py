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
        (DecomonConv2D, dict(filters=10, kernel_size=(3, 3), data_format="channels_last")),
        (DecomonReshape, dict(target_shape=(1, -1, 1))),
        (Reshape, dict(target_shape=(1, -1, 1))),
        (DecomonGroupSort2, dict()),
    ],
)
@pytest.mark.parametrize(
    "kerastensor_inputs_fn_name, kerastensor_inputs_kwargs, np_inputs_fn_name, np_inputs_kwargs",
    [
        ("get_tensor_decomposition_1d_box", dict(), "get_standard_values_1d_box", dict(n=0)),
        ("get_tensor_decomposition_multid_box", dict(odd=0), "get_standard_values_multid_box", dict(odd=0)),
        (
            "get_tensor_decomposition_images_box",
            dict(odd=0, data_format="channels_last"),
            "get_standard_values_images_box",
            dict(odd=0, data_format="channels_last"),
        ),
    ],
)
def test_compute_output_shape(
    helpers,
    mode,
    layer_class,
    layer_kwargs,
    kerastensor_inputs_fn_name,
    kerastensor_inputs_kwargs,
    np_inputs_fn_name,
    np_inputs_kwargs,
):
    dc_decomp = False
    if (layer_class == DecomonBatchNormalization or layer_class == DecomonMaxPooling2D) and dc_decomp:
        pytest.skip(f"{layer_class} with dc_decomp=True not yet implemented.")
    if (
        layer_class == DecomonConv2D or layer_class == DecomonMaxPooling2D
    ) and kerastensor_inputs_fn_name != "get_tensor_decomposition_images_box":
        pytest.skip(f"{layer_class} applies only on image-like inputs.")
    if layer_class == DecomonGroupSort2:
        if not deel_lip_available:
            pytest.skip("deel-lip is not available")

    # contruct inputs functions
    kerastensor_inputs_fn = getattr(helpers, kerastensor_inputs_fn_name)
    kerastensor_inputs_kwargs["dc_decomp"] = dc_decomp
    np_inputs_fn = getattr(helpers, np_inputs_fn_name)
    np_inputs_kwargs["dc_decomp"] = dc_decomp

    # tensors inputs
    inputs = kerastensor_inputs_fn(**kerastensor_inputs_kwargs)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    inputs_ref = helpers.get_input_ref_from_full_inputs(inputs)

    # numpy inputs
    inputs_ = np_inputs_fn(**np_inputs_kwargs)
    inputs_for_mode_ = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_, mode=mode, dc_decomp=dc_decomp)

    # construct and build original layer (decomon or keras)
    if issubclass(layer_class, DecomonLayer):
        layer = layer_class(mode=mode, dc_decomp=dc_decomp, **layer_kwargs)
        layer(inputs_for_mode)
    else:  # keras layer
        layer = layer_class(**layer_kwargs)
        layer(inputs_ref)

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
