import pytest

from decomon.layers.decomon_layers import (
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
        (DecomonConv2D, dict(filters=10, kernel_size=(3, 3), data_format="channels_last")),
        (DecomonMaxPooling2D, dict(pool_size=(2, 2), strides=(2, 2), padding="valid")),
        (DecomonReshape, dict(target_shape=(1, -1, 1))),
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
    dc_decomp,
    layer_class,
    layer_kwargs,
    kerastensor_inputs_fn_name,
    kerastensor_inputs_kwargs,
    np_inputs_fn_name,
    np_inputs_kwargs,
):
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

    # numpy inputs
    inputs_ = np_inputs_fn(**np_inputs_kwargs)
    inputs_for_mode_ = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_, mode=mode, dc_decomp=dc_decomp)

    # construct layer
    layer = layer_class(mode=mode, dc_decomp=dc_decomp, **layer_kwargs)

    # check symbolic tensor output shapes
    inputshapes = [i.shape for i in inputs_for_mode]
    outputshapes = layer.compute_output_shape(inputshapes)
    outputs = layer(inputs_for_mode)
    assert [o.shape for o in outputs] == outputshapes

    # check output shapes for concrete call
    outputs_ = layer(inputs_for_mode_)
    # compare without batch sizes
    assert [tuple(o.shape)[1:] for o in outputs_] == [s[1:] for s in outputshapes]


@pytest.mark.parametrize(
    "layer_class, layer_kwargs",
    [
        (DecomonAdd, dict()),
        (DecomonConcatenate, dict()),
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
def test_merge_layers_compute_output_shape(
    helpers,
    mode,
    dc_decomp,
    layer_class,
    layer_kwargs,
    kerastensor_inputs_fn_name,
    kerastensor_inputs_kwargs,
    np_inputs_fn_name,
    np_inputs_kwargs,
):
    if dc_decomp:
        pytest.skip(f"{layer_class} with dc_decomp=True not yet implemented.")

    # contruct inputs functions
    kerastensor_inputs_fn = getattr(helpers, kerastensor_inputs_fn_name)
    kerastensor_inputs_kwargs["dc_decomp"] = dc_decomp
    np_inputs_fn = getattr(helpers, np_inputs_fn_name)
    np_inputs_kwargs["dc_decomp"] = dc_decomp

    # tensors inputs
    inputs_1 = kerastensor_inputs_fn(**kerastensor_inputs_kwargs)
    inputs_for_mode_1 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1, mode=mode, dc_decomp=dc_decomp)
    inputs_2 = kerastensor_inputs_fn(**kerastensor_inputs_kwargs)
    inputs_for_mode_2 = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_2, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode = inputs_for_mode_1 + inputs_for_mode_2

    # numpy inputs
    inputs_1_ = np_inputs_fn(**np_inputs_kwargs)
    inputs_for_mode_1_ = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_1_, mode=mode, dc_decomp=dc_decomp)
    inputs_2_ = np_inputs_fn(**np_inputs_kwargs)
    inputs_for_mode_2_ = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_2_, mode=mode, dc_decomp=dc_decomp)
    inputs_for_mode_ = inputs_for_mode_1_ + inputs_for_mode_2_

    # construct layer
    layer = layer_class(mode=mode, dc_decomp=dc_decomp, **layer_kwargs)

    # check symbolic tensor output shapes
    inputshapes = [i.shape for i in inputs_for_mode]
    outputshapes = layer.compute_output_shape(inputshapes)
    outputs = layer(inputs_for_mode)
    assert [o.shape for o in outputs] == outputshapes

    # check output shapes for concrete call
    outputs_ = layer(inputs_for_mode_)
    # compare without batch sizes
    assert [tuple(o.shape)[1:] for o in outputs_] == [s[1:] for s in outputshapes]
