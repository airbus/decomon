import keras.config as keras_config
import numpy as np
import pytest
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    Permute,
    Reshape,
)
from keras.ops import convert_to_numpy
from keras.src.layers.merging.dot import batch_dot

from decomon.backward_layers.convert import to_backward
from decomon.core import ForwardMode
from decomon.layers.core import DecomonLayer
from decomon.layers.decomon_layers import DecomonDense


@pytest.mark.parametrize(
    "layer_class, layer_kwargs",
    [
        (Dense, dict(units=3, use_bias=True)),
        (Dense, dict(units=3, use_bias=False)),
        (Activation, dict(activation="linear")),
        (Activation, dict(activation="relu")),
        (Reshape, dict(target_shape=(1, -1, 1))),
        (BatchNormalization, dict()),
        (BatchNormalization, dict(axis=1)),
        (BatchNormalization, dict(center=False, scale=False)),
        (BatchNormalization, dict(axis=2, center=False)),
        (BatchNormalization, dict(axis=3)),
        (Conv2D, dict(filters=10, kernel_size=(3, 3), data_format="channels_last")),
        (Flatten, dict()),
        (Dropout, dict(rate=0.9)),
        (Permute, dict(dims=(1,))),
        (Permute, dict(dims=(2, 1, 3))),
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
@pytest.mark.parametrize("randomize_weights", [False, True])
def test_backward_layer(
    helpers,
    mode,
    layer_class,
    layer_kwargs,
    kerastensor_inputs_fn_name,
    kerastensor_inputs_kwargs,
    np_inputs_fn_name,
    np_inputs_kwargs,
    randomize_weights,
):
    # skip nonsensical combinations
    if (
        layer_class is BatchNormalization
        and "axis" in layer_kwargs
        and layer_kwargs["axis"] > 1
        and kerastensor_inputs_fn_name != "get_tensor_decomposition_images_box"
    ):
        pytest.skip("batchnormalization on axis>1 possible only for image-like data")
    if layer_class in (Conv2D, MaxPooling2D) and kerastensor_inputs_fn_name != "get_tensor_decomposition_images_box":
        pytest.skip("convolution and maxpooling2d possible only for image-like data")
    if (
        layer_class is Permute
        and len(layer_kwargs["dims"]) < 3
        and kerastensor_inputs_fn_name == "get_tensor_decomposition_images_box"
    ):
        pytest.skip("1d permutation not possible for image-like data")
    if (
        layer_class is Permute
        and len(layer_kwargs["dims"]) == 3
        and kerastensor_inputs_fn_name != "get_tensor_decomposition_images_box"
    ):
        pytest.skip("3d permutation possible only for image-like data")

    dc_decomp = False
    decimal = 4

    # contruct inputs functions
    kerastensor_inputs_fn = getattr(helpers, kerastensor_inputs_fn_name)
    kerastensor_inputs_kwargs["dc_decomp"] = dc_decomp
    np_inputs_fn = getattr(helpers, np_inputs_fn_name)
    np_inputs_kwargs["dc_decomp"] = dc_decomp

    # tensors inputs
    inputs = kerastensor_inputs_fn(**kerastensor_inputs_kwargs)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs)

    # numpy inputs
    inputs_ = np_inputs_fn(**np_inputs_kwargs)
    inputs_for_mode_ = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs_, mode=mode, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)

    # construct and build original layer (keras)
    layer = layer_class(**layer_kwargs)
    output_ref = layer(input_ref)
    f_ref = helpers.function(inputs, output_ref)

    # randomize weights
    if randomize_weights:
        for weight in layer.weights:
            weight.assign(np.random.random(tuple(weight.shape)))

    # get backward layer & function
    backward_layer = to_backward(layer, mode=mode)
    outputs = backward_layer(inputs_for_mode)
    f_backward = helpers.function(inputs, outputs)

    # keras & decomon outputs
    output_ref_ = f_ref(inputs_)
    outputs_ = f_backward(inputs_)

    # flattened keras inputs/outputs
    input_ref_flat_ = np.reshape(input_ref_, (input_ref_.shape[0], -1))
    output_ref_flat_ = np.reshape(output_ref_, (output_ref_.shape[0], -1))

    # check bounds consistency
    w_u_out_, b_u_out_, w_l_out_, b_l_out_ = outputs_
    upper_ = convert_to_numpy(batch_dot(w_u_out_, input_ref_flat_, axes=(-2, -1))) + b_u_out_  # batch mult
    lower_ = convert_to_numpy(batch_dot(w_l_out_, input_ref_flat_, axes=(-2, -1))) + b_l_out_  # batch mult
    np.testing.assert_almost_equal(
        np.clip(lower_ - output_ref_flat_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_min >x_"
    )
    np.testing.assert_almost_equal(
        np.clip(output_ref_flat_ - upper_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_max < x_"
    )
