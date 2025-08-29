import keras.ops as K
import numpy as np
import pytest
from keras.layers import (
    Activation,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    LeakyReLU,
    MaxPooling2D,
    Permute,
    ReLU,
    RepeatVector,
    Reshape,
    UpSampling1D,
    UpSampling2D,
    UpSampling3D,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
)
from keras_custom.layers import Max, Min, MulConstant
from pytest_cases import (
    fixture,
    fixture_union,
    param_fixture,
    parametrize,
    unpack_fixture,
)

from decomon.constants import Slope
from decomon.keras_utils import batch_multid_dot
from decomon.layers import (
    DecomonActivation,
    DecomonAveragePooling1D,
    DecomonAveragePooling2D,
    DecomonAveragePooling3D,
    DecomonBatchNormalization,
    DecomonConv1D,
    DecomonConv2D,
    DecomonConv3D,
    DecomonCropping1D,
    DecomonCropping2D,
    DecomonCropping3D,
    DecomonDense,
    DecomonDepthwiseConv1D,
    DecomonDepthwiseConv2D,
    DecomonDropout,
    DecomonFlatten,
    DecomonGlobalAveragePooling1D,
    DecomonGlobalAveragePooling2D,
    DecomonGlobalAveragePooling3D,
    DecomonLeakyReLU,
    DecomonMax,
    DecomonMaxPooling2D,
    DecomonMin,
    DecomonMulConstant,
    DecomonPermute,
    DecomonReLU,
    DecomonRepeatVector,
    DecomonReshape,
    DecomonUpSampling1D,
    DecomonUpSampling2D,
    DecomonUpSampling3D,
    DecomonZeroPadding1D,
    DecomonZeroPadding2D,
    DecomonZeroPadding3D,
)
from decomon.layers.activations.activation import DecomonLinear


@fixture
def dense_keras_kwargs(use_bias):
    return dict(units=7, use_bias=use_bias)


activation_with_slope = param_fixture(
    "activation_with_slope",
    [
        "relu",
        "exponential",
        "elu",
        "leaky_relu",
        "selu",
        "softplus",
    ],
)
activation_without_slope = param_fixture(
    "activation_without_slope",
    [
        None,
        "softsign",
        "sigmoid",
        "tanh",
    ],
)


@fixture
def activation_without_slope_kwargs(activation_without_slope):
    return _activation_kwargs(activation=activation_without_slope)


@fixture
def activation_with_slope_kwargs(activation_with_slope, slope):
    # skip adaptative slope for non relu activation (not yet implemented)
    if Slope(slope) == Slope.A_SLOPE:
        if activation_with_slope != "relu":
            pytest.skip("adaptative slope not implemented for non-relu activations")

    return _activation_kwargs(activation=activation_with_slope, slope=slope)


def _activation_kwargs(activation, slope=None):
    keras_kwargs = dict(activation=activation)
    if activation == "relu":
        decomon_kwargs = dict(slope=slope)
    else:
        decomon_kwargs = {}
    return keras_kwargs, decomon_kwargs


activation_kwargs = fixture_union(
    "activation_kwargs",
    [activation_without_slope_kwargs, activation_with_slope_kwargs],
)
activation_keras_kwargs, activation_decomon_kwargs = unpack_fixture(
    "activation_keras_kwargs, activation_decomon_kwargs", activation_kwargs
)


@fixture
def data_format_kwargs(data_format):
    return dict(data_format=data_format)


@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs",
    [
        (DecomonDense, {}, Dense, dense_keras_kwargs),
        (DecomonActivation, activation_decomon_kwargs, Activation, activation_keras_kwargs),
        (DecomonZeroPadding2D, {}, ZeroPadding2D, dict(padding=((1, 3), (0, 5)))),
        (DecomonCropping2D, {}, Cropping2D, data_format_kwargs),
        (DecomonUpSampling2D, {}, UpSampling2D, data_format_kwargs),
        (DecomonReshape, {}, Reshape, dict(target_shape=(2, -1))),
        (DecomonRepeatVector, {}, RepeatVector, dict(n=2)),
        (DecomonPermute, {}, Permute, dict(dims=(2, 3, 1))),
        (DecomonFlatten, {}, Flatten, dict()),
        (DecomonDropout, {}, Dropout, dict(rate=0.2)),
        # (DecomonMaxPooling2D, {}, MaxPooling2D, data_format_kwargs),  # error with diagonal entries
        (DecomonAveragePooling2D, {}, AveragePooling2D, dict(pool_size=2)),
        (DecomonGlobalAveragePooling2D, {}, GlobalAveragePooling2D, data_format_kwargs),
        (DecomonGlobalAveragePooling2D, {}, GlobalAveragePooling2D, data_format_kwargs),
        (DecomonBatchNormalization, {}, BatchNormalization, dict()),
        (DecomonBatchNormalization, {}, BatchNormalization, dict(center=False, scale=False)),
        (DecomonConv2D, {}, Conv2D, dict(filters=2, kernel_size=2)),
        (DecomonDepthwiseConv2D, {}, DepthwiseConv2D, dict(kernel_size=2)),
        (DecomonMax, {}, Max, dict(axis=1, keepdims=False)),
        (DecomonMax, {}, Max, dict(axis=1, keepdims=True)),
        # (DecomonMulConstant, {}, MulConstant, dict(constant=3.14)),  # to be fixed
        (DecomonMin, {}, Min, dict(axis=1, keepdims=False)),
        (DecomonMin, {}, Min, dict(axis=1, keepdims=True)),
        (DecomonMin, {}, Min, dict(axis=-1, keepdims=False)),
        (DecomonMin, {}, Min, dict(axis=-1, keepdims=True)),
        (DecomonLeakyReLU, {}, LeakyReLU, {}),
        (DecomonReLU, {}, ReLU, {}),
    ],
)
def test_decomon_unary_layer(
    decomon_layer_class,
    decomon_layer_kwargs,
    keras_layer_class,
    keras_layer_kwargs,
    ibp,
    affine,
    propagation,
    perturbation_domain,
    batchsize,
    keras_symbolic_model_input_fn,
    keras_symbolic_layer_input_fn,
    decomon_symbolic_input_fn,
    keras_model_input_fn,
    keras_layer_input_fn,
    decomon_input_fn,
    equal_ibp_bounds,
    equal_affine_bounds,
    helpers,
):
    decimal = 4

    # symbolic inputs for keras layer and keras model (supposed to contain the layer)
    keras_symbolic_model_input = keras_symbolic_model_input_fn()
    keras_symbolic_layer_input = keras_symbolic_layer_input_fn(keras_symbolic_model_input)

    # init keras layer
    layer = keras_layer_class(**keras_layer_kwargs)

    # skip some cases where the input shape is incompatible with the layer
    keras_layer_class_for_4dinputs_only = {
        ZeroPadding2D,
        Cropping2D,
        MaxPooling2D,
        UpSampling2D,
        AveragePooling2D,
        GlobalAveragePooling2D,
        Conv2D,
        DepthwiseConv2D,
    }
    if keras_layer_class in keras_layer_class_for_4dinputs_only:
        if len(keras_symbolic_layer_input.shape) != 4:
            pytest.skip(f"{keras_layer_class.__name__} works only with 4D inputs")
    if isinstance(layer, Reshape):
        if np.prod(keras_symbolic_layer_input.shape[1:]) % 2 != 0:
            pytest.skip("reshaping only even shaped inputs in this test")

    if isinstance(layer, RepeatVector):
        if len(keras_symbolic_layer_input.shape) != 2:
            pytest.skip("RepeatVector works only with 2D inputs")
    if isinstance(layer, Permute):
        if len(keras_symbolic_layer_input.shape) != 4:
            pytest.skip("test Permute in 4D only")
    if isinstance(layer, Max) or isinstance(layer, Min):
        if len(keras_symbolic_layer_input.shape) <= 2 and not layer.keepdims:
            pytest.skip("test Max for 0d/1d input only with keepdims=True")

    # build keras layer
    layer(keras_symbolic_layer_input)

    # randomize weights (e.g. to test non-zero biases)
    for w in layer.weights:
        # positive-only weights ?
        if "variance" in w.name:  # like BatchNormalization.moving_variance
            w.assign(np.random.random(w.shape) + 0.5)  # between 0.5 and 1.5
        else:
            w.assign(2.0 * np.random.random(w.shape) - 1.0)  # between -1 and 1

    # init + build decomon layer
    output_shape = layer.output.shape[1:]
    model_output_shape = output_shape
    model_input_shape = keras_symbolic_model_input.shape[1:]
    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape=model_output_shape,
        model_input_shape=model_input_shape,
        **decomon_layer_kwargs,
    )

    decomon_symbolic_inputs = decomon_symbolic_input_fn(output_shape=output_shape, linear=decomon_layer.linear)
    decomon_layer(decomon_symbolic_inputs)

    # call on actual inputs
    keras_model_input = keras_model_input_fn()
    keras_layer_input = keras_layer_input_fn(keras_model_input)
    decomon_inputs = decomon_input_fn(
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        output_shape=output_shape,
        linear=decomon_layer.linear,
    )

    keras_output = layer(keras_layer_input)
    decomon_output = decomon_layer(decomon_inputs)

    # check affine representation is ok  (except for linear activation, undefined)
    if decomon_layer.linear and not (
        (isinstance(decomon_layer, DecomonActivation) and isinstance(decomon_layer.decomon_activation, DecomonLinear))
        or isinstance(decomon_layer, DecomonLinear)
    ):
        w, b = decomon_layer.get_affine_representation()
        diagonal = (False, w.shape == b.shape)
        missing_batchsize = (False, True)
        keras_output_2 = (
            batch_multid_dot(keras_layer_input, w, missing_batchsize=missing_batchsize, diagonal=diagonal) + b
        )
        np.testing.assert_almost_equal(
            K.convert_to_numpy(keras_output),
            K.convert_to_numpy(keras_output_2),
            decimal=decimal,
            err_msg="wrong affine representation",
        )

    # check output shapes
    input_shape = [t.shape for t in decomon_inputs]
    output_shape = [t.shape for t in decomon_output]
    expected_output_shape = decomon_layer.compute_output_shape(input_shape)
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_layer(
        decomon_output=decomon_output,
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        keras_model_output=keras_output,
        keras_layer_output=keras_output,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        decimal=decimal,
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    if decomon_layer_class.linear:
        helpers.assert_decomon_output_lower_equal_upper(
            decomon_output,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            decimal=decimal,
            check_ibp=equal_ibp_bounds,
            check_affine=equal_affine_bounds,
        )


@parametrize(
    "decomon_layer_class, decomon_layer_kwargs, keras_layer_class, keras_layer_kwargs, input_shape_wo_batchsize",
    [
        (DecomonZeroPadding1D, {}, ZeroPadding1D, dict(padding=(0, 5)), (2, 3)),
        (DecomonZeroPadding3D, {}, ZeroPadding3D, dict(padding=2), (1, 2, 2, 3)),
        (DecomonCropping1D, {}, Cropping1D, {}, (3, 2)),
        (DecomonCropping3D, {}, Cropping3D, dict(cropping=((0, 1), (2, 2), (1, 0))), (2, 5, 3, 2)),
        (
            DecomonCropping3D,
            {},
            Cropping3D,
            dict(data_format="channels_first", cropping=((0, 1), (2, 2), (1, 0))),
            (2, 2, 5, 3),
        ),
        (DecomonUpSampling1D, {}, UpSampling1D, {}, (1, 2)),
        (DecomonUpSampling3D, {}, UpSampling3D, data_format_kwargs, (2, 2, 2, 2)),
        (DecomonGlobalAveragePooling1D, {}, GlobalAveragePooling1D, data_format_kwargs, (3, 2)),
        (DecomonGlobalAveragePooling3D, {}, GlobalAveragePooling3D, data_format_kwargs, (2, 3, 2, 3)),
        (DecomonAveragePooling1D, {}, AveragePooling1D, dict(pool_size=2), (5, 1)),
        # (DecomonAveragePooling3D, {}, AveragePooling3D, dict(pool_size=(2,1,1)), (5,2,2,1)),  # expected scalar type Float but found Double
        (DecomonConv1D, {}, Conv1D, dict(filters=2, kernel_size=2), (5, 3)),
        (DecomonConv3D, {}, Conv3D, dict(filters=2, kernel_size=2), (3, 3, 3, 2)),
        (DecomonDepthwiseConv1D, {}, DepthwiseConv1D, dict(kernel_size=2), (5, 2)),
    ],
)
def test_decomon_unary_layer_specific_shapes(
    decomon_layer_class,
    decomon_layer_kwargs,
    keras_layer_class,
    keras_layer_kwargs,
    input_shape_wo_batchsize,
    ibp,
    affine,
    propagation,
    perturbation_domain,
    batchsize,
    simple_layer_input_functions_from_input_shape_wo_batchsize,
    helpers,
):
    decimal = 4
    (
        keras_symbolic_model_input_fn,
        keras_symbolic_layer_input_fn,
        decomon_symbolic_input_fn,
        keras_model_input_fn,
        keras_layer_input_fn,
        decomon_input_fn,
        equal_ibp_bounds,
        equal_affine_bounds,
    ) = simple_layer_input_functions_from_input_shape_wo_batchsize

    # symbolic inputs for keras layer and keras model (supposed to contain the layer)
    keras_symbolic_model_input = keras_symbolic_model_input_fn(input_shape_wo_batchsize)
    keras_symbolic_layer_input = keras_symbolic_layer_input_fn(input_shape_wo_batchsize, keras_symbolic_model_input)

    # init keras layer
    layer = keras_layer_class(**keras_layer_kwargs)

    # build keras layer
    layer(keras_symbolic_layer_input)

    # randomize weights between -1 and 1 => non-zero biases
    for w in layer.weights:
        w.assign(2.0 * np.random.random(w.shape) - 1.0)

    # init + build decomon layer
    output_shape = layer.output.shape[1:]
    model_output_shape = output_shape
    model_input_shape = keras_symbolic_model_input.shape[1:]

    decomon_layer = decomon_layer_class(
        layer=layer,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        model_output_shape=model_output_shape,
        model_input_shape=model_input_shape,
        **decomon_layer_kwargs,
    )

    decomon_symbolic_inputs = decomon_symbolic_input_fn(
        input_shape_wo_batchsize, output_shape=output_shape, linear=decomon_layer.linear
    )
    decomon_layer(decomon_symbolic_inputs)

    # call on actual inputs
    keras_model_input = keras_model_input_fn(input_shape_wo_batchsize)
    keras_layer_input = keras_layer_input_fn(input_shape_wo_batchsize, keras_model_input)
    decomon_inputs = decomon_input_fn(
        input_shape_wo_batchsize,
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        output_shape=output_shape,
        linear=decomon_layer.linear,
    )

    keras_output = layer(keras_layer_input)
    decomon_output = decomon_layer(decomon_inputs)

    # check affine representation is ok  (except for linear activation, undefined)
    if decomon_layer.linear and not (
        (isinstance(decomon_layer, DecomonActivation) and isinstance(decomon_layer.decomon_activation, DecomonLinear))
        or isinstance(decomon_layer, DecomonLinear)
    ):
        w, b = decomon_layer.get_affine_representation()
        diagonal = (False, w.shape == b.shape)
        missing_batchsize = (False, True)
        keras_output_2 = (
            batch_multid_dot(keras_layer_input, w, missing_batchsize=missing_batchsize, diagonal=diagonal) + b
        )
        np.testing.assert_almost_equal(
            K.convert_to_numpy(keras_output),
            K.convert_to_numpy(keras_output_2),
            decimal=decimal,
            err_msg="wrong affine representation",
        )

    # check output shapes
    input_shape = [t.shape for t in decomon_inputs]
    output_shape = [t.shape for t in decomon_output]
    expected_output_shape = decomon_layer.compute_output_shape(input_shape)
    expected_output_shape = helpers.replace_none_by_batchsize(shapes=expected_output_shape, batchsize=batchsize)
    assert output_shape == expected_output_shape

    # check ibp and affine bounds well ordered w.r.t. keras inputs/outputs
    helpers.assert_decomon_output_compare_with_keras_input_output_layer(
        decomon_output=decomon_output,
        keras_model_input=keras_model_input,
        keras_layer_input=keras_layer_input,
        keras_model_output=keras_output,
        keras_layer_output=keras_output,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        decimal=decimal,
    )

    # before propagation through linear layer lower == upper => lower == upper after propagation
    if decomon_layer_class.linear:
        helpers.assert_decomon_output_lower_equal_upper(
            decomon_output,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            decimal=decimal,
            check_ibp=equal_ibp_bounds,
            check_affine=equal_affine_bounds,
        )
