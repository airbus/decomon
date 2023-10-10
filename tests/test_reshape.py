import numpy as np
import pytest
from keras_core.layers import Permute, Reshape

from decomon.core import ForwardMode, get_affine, get_ibp
from decomon.layers.convert import to_decomon
from decomon.layers.decomon_reshape import DecomonPermute, DecomonReshape


def keras_target_shape_reshape(input_ref):
    return (np.prod(input_ref.shape[1:]),)


def target_shape_keras2np_reshape(target_shape):
    return (-1, target_shape[0])


def keras_target_shape_permute(input_ref):
    n_dim = len(input_ref.shape) - 1
    return np.random.permutation(n_dim) + 1


def target_shape_keras2np_permute(target_shape):
    return tuple([0] + list(target_shape))


@pytest.mark.parametrize(
    "decomon_layer_class, keras_target_shape_func, target_shape_keras2np_func, np_func",
    [
        (DecomonReshape, keras_target_shape_reshape, target_shape_keras2np_reshape, np.reshape),
        (DecomonPermute, keras_target_shape_permute, target_shape_keras2np_permute, np.transpose),
    ],
)
def test_Decomon_reshape_n_permute_box(
    decomon_layer_class,
    keras_target_shape_func,
    target_shape_keras2np_func,
    np_func,
    mode,
    dc_decomp,
    floatx,
    decimal,
    helpers,
):
    odd, m_0, m_1 = 0, 0, 1
    data_format = "channels_last"

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs=inputs_)

    # target shape
    target_shape = keras_target_shape_func(input_ref)
    target_shape_ = target_shape_keras2np_func(target_shape)

    # original output
    output_ref_ = np_func(input_ref_, target_shape_)

    # decomon layer
    decomon_layer = decomon_layer_class(target_shape, dc_decomp=dc_decomp, mode=mode, dtype=K.floatx())

    # decomon output
    output = decomon_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, output)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


@pytest.mark.parametrize(
    "keras_target_shape_func, keras_layer_class",
    [
        (keras_target_shape_reshape, Reshape),
        (keras_target_shape_permute, Permute),
    ],
)
def test_Decomon_reshape_n_permute_to_decomon_box(
    keras_target_shape_func, keras_layer_class, shared, floatx, decimal, helpers
):
    odd, m_0, m_1 = 0, 0, 1
    dc_decomp = True
    data_format = "channels_last"
    mode = ForwardMode.HYBRID
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)

    #  target shape
    target_shape = keras_target_shape_func(input_ref)

    # keras layer
    keras_layer = keras_layer_class(target_shape, dtype=K.floatx())

    # original output
    output_ref = keras_layer(input_ref)
    f_ref = helpers.function(inputs, output_ref)
    output_ref_ = f_ref(inputs_)

    # conversion with to_decomon
    input_dim = helpers.get_input_dim_from_full_inputs(inputs_)
    decomon_layer = to_decomon(keras_layer, input_dim, dc_decomp=dc_decomp, shared=shared, ibp=ibp, affine=affine)

    # decomon outputs
    outputs = decomon_layer(inputs_for_mode)
    f_decomon = helpers.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)

    # check bounds consistency
    helpers.assert_decomon_layer_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        dc_decomp=dc_decomp,
        mode=mode,
        decimal=decimal,
    )
