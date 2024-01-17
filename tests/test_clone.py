# creating toy network and assess that the decomposition is correct


import keras.config as keras_config
import keras.ops as K
import numpy as np
import pytest
from keras.layers import Activation, Dense, Flatten, Input, Reshape
from keras.models import Model, Sequential

from decomon.core import ForwardMode, Slope, get_affine, get_ibp, get_mode
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.models import clone
from decomon.models.convert import FeedDirection, get_direction
from decomon.models.utils import (
    ConvertMethod,
    get_ibp_affine_from_method,
    has_merge_layers,
)


def test_convert_nok_several_inputs():
    a = Input((1,))
    b = Input((2,))
    model = Model([a, b], a)

    with pytest.raises(ValueError, match="only 1 input"):
        clone(model)


def test_convert_nok_unflattened_input():
    a = Input((1, 2))
    model = Model(a, a)

    with pytest.raises(ValueError, match="flattened input"):
        clone(model)


def test_clone_with_backbounds(method, helpers):
    dc_decomp = False
    n = 0
    ibp, affine = get_ibp_affine_from_method(method)
    mode = get_mode(ibp, affine)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)
    input_decomon_wo_backbounds = helpers.get_inputs_np_for_decomon_model_from_full_inputs(inputs=inputs_)

    #  keras model and output of reference
    ref_nn = helpers.toy_network_tutorial()
    output_ref_ = helpers.predict_on_small_numpy(ref_nn, input_ref_)

    # create back_bounds for adversarial robustness like studies
    output_dim = int(np.prod(ref_nn.output_shape[1:]))
    C = Input((output_dim, output_dim))
    batchsize = input_decomon_wo_backbounds.shape[0]
    C_ = np.repeat(np.identity(output_dim), repeats=batchsize, axis=0)
    inputs_decomon_ = [input_decomon_wo_backbounds, C_]

    # decomon conversion
    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, back_bounds=[C])

    #  decomon outputs
    outputs_ = helpers.predict_on_small_numpy(decomon_model, inputs_decomon_)

    #  check bounds consistency
    helpers.assert_decomon_model_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
    )


def test_convert_1D(n, method, mode, floatx, decimal, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)
    input_decomon_ = helpers.get_inputs_np_for_decomon_model_from_full_inputs(inputs=inputs_)

    #  keras model and output of reference
    ref_nn = helpers.toy_network_tutorial(dtype=keras_config.floatx())
    output_ref_ = helpers.predict_on_small_numpy(ref_nn, input_ref_)

    # decomon conversion
    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)

    #  decomon outputs
    outputs_ = helpers.predict_on_small_numpy(decomon_model, input_decomon_)

    #  check bounds consistency
    helpers.assert_decomon_model_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


def test_convert_1D_forward_slope(slope, helpers):
    ibp = True
    affine = True
    dc_decomp = False

    n, method, mode = 0, "forward-hybrid", "hybrid"
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    ref_nn = helpers.toy_network_tutorial(dtype=keras_config.floatx())
    ref_nn(input_ref)

    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of activation layers
    for layer in decomon_model.layers:
        if layer.__class__.__name__.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_convert_1D_backward_slope(slope, helpers):
    n, method, mode = 0, "crown-forward-hybrid", "hybrid"
    ibp = True
    affine = True
    dc_decomp = False

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    ref_nn = helpers.toy_network_tutorial(dtype=keras_config.floatx())
    ref_nn(input_ref)

    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of layers with activation
    for layer in decomon_model.layers:
        layer_class_name = layer.__class__.__name__
        if layer_class_name.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_name_forward():
    layers = []
    layers.append(Input((1,)))
    layers.append(Dense(1))
    layers.append(Dense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(Dense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_f = clone(model=model, method=ConvertMethod.FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 1
    nb_batman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 2


def test_name_backward():
    layers = []
    layers.append(Input((1,)))
    layers.append(Dense(1))
    layers.append(Dense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(Dense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_b = clone(model=model, method=ConvertMethod.CROWN_FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 2
    nb_batman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 3


def test_convert_toy_models_1d(toy_model_1d, method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    decimal = 4
    n = 6
    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)
    input_decomon_ = helpers.get_inputs_np_for_decomon_model_from_full_inputs(inputs=inputs_)

    #  keras model and output of reference
    ref_nn = toy_model_1d
    output_ref_ = helpers.predict_on_small_numpy(ref_nn, input_ref_)

    # decomon conversion
    if (get_direction(method) == FeedDirection.BACKWARD) and has_merge_layers(ref_nn):
        # skip models with merge layers in backward direction as not yet implemented
        with pytest.raises(NotImplementedError):
            decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)
        return

    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)

    #  decomon outputs
    outputs_ = helpers.predict_on_small_numpy(decomon_model, input_decomon_)

    #  check bounds consistency
    helpers.assert_decomon_model_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


def test_convert_cnn(method, mode, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    if get_direction(method) == FeedDirection.BACKWARD:
        # skip as BackwardConv2D not yet ready
        pytest.skip(f"BackwardConv2D not yet fully implemented")

    decimal = 4
    data_format = "channels_last"
    odd, m_0, m_1 = 0, 0, 1

    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)
    input_ref_min_, input_ref_max_ = helpers.get_input_ref_bounds_from_full_inputs(inputs_)

    # flatten inputs
    preprocess_layer = Flatten(data_format=data_format)
    input_ref_reshaped_ = K.convert_to_numpy(preprocess_layer(input_ref_))
    input_ref_min_reshaped_ = K.convert_to_numpy(preprocess_layer(input_ref_min_))
    input_ref_max_reshaped_ = K.convert_to_numpy(preprocess_layer(input_ref_max_))

    input_decomon_ = np.concatenate((input_ref_min_reshaped_[:, None], input_ref_max_reshaped_[:, None]), axis=1)

    #  keras model and output of reference
    image_data_shape = input_ref_.shape[1:]  # image shape: before flattening
    ref_nn = helpers.toy_struct_cnn(dtype=keras_config.floatx(), image_data_shape=image_data_shape)
    output_ref_ = helpers.predict_on_small_numpy(ref_nn, input_ref_reshaped_)

    # decomon conversion
    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)

    #  decomon outputs
    outputs_ = helpers.predict_on_small_numpy(decomon_model, input_decomon_)

    #  check bounds consistency
    z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = helpers.get_full_outputs_from_outputs_for_mode(
        outputs_for_mode=outputs_, mode=mode, dc_decomp=dc_decomp, full_inputs=inputs_
    )
    helpers.assert_output_properties_box(
        input_ref_reshaped_,
        output_ref_,
        h_,
        g_,
        input_ref_min_reshaped_,
        input_ref_max_reshaped_,
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        decimal=decimal,
    )
