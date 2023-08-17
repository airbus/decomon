# Test unit for decomon with Dense layers
import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.python.keras.backend import _get_available_gpus

from decomon.backward_layers.convert import to_backward
from decomon.layers.maxpooling_opt import DecomonMaxPooling2D
from decomon.layers.convert import to_decomon
from decomon.core import ForwardMode, get_affine, get_ibp


def test_Decomon_pool_box(data_format, padding, use_bias, mode, floatx, decimal, helpers):
    if data_format == "channels_first" and not len(_get_available_gpus()):
        pytest.skip("data format 'channels first' is possible only in GPU mode")

    odd, m_0, m_1 = 0, 0, 1
    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    kwargs_layer = dict(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=data_format, dtype=K.floatx())

    # tensor inputs
    inputs = helpers.get_tensor_decomposition_images_box(data_format, odd, dc_decomp=dc_decomp)
    inputs_for_mode = helpers.get_inputs_for_mode_from_full_inputs(inputs=inputs, mode=mode, dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    # numpy inputs
    inputs_ = helpers.get_standard_values_images_box(data_format, odd, m0=m_0, m1=m_1, dc_decomp=dc_decomp)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    # decomon layer
    keras_layer = MaxPooling2D(**kwargs_layer)
    output_ref = keras_layer(input_ref)
    input_dim = helpers.get_input_dim_from_full_inputs(inputs)
    decomon_layer = to_decomon(keras_layer, input_dim, dc_decomp=dc_decomp, shared=True, ibp=ibp, affine=affine)
    _ = decomon_layer(inputs_for_mode)

    # get backward layer
    backward_layer = to_backward(decomon_layer)
    # backward outputs
    outputs = backward_layer(inputs_for_mode)
    f_decomon = K.function(inputs, outputs)
    outputs_ = f_decomon(inputs_)
    output_ref_ = K.function(inputs, output_ref)(inputs_)
    # flatten output_ref_
    output_ref_ = np.reshape(output_ref_, (len(output_ref_), -1))

    w_u_, b_u_, w_l_, b_l_ = outputs_

    # reshape the matrices
    w_u_ = w_u_[:, None]
    w_l_ = w_l_[:, None]

    b_l = b_l.reshape((len(b_l), -1))
    b_u = b_u.reshape((len(b_u), -1))
    W_u = W_u.reshape((len(W_u), W_u.shape[1], -1))
    W_l = W_l.reshape((len(W_l), W_l.shape[1], -1))

    # backward recomposition
    w_r_u = np.sum(np.maximum(0.0, w_u_) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(0.0, w_u_) * np.expand_dims(W_l, -1), 2
    )
    w_r_l = np.sum(np.maximum(0.0, w_l_) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(0.0, w_l_) * np.expand_dims(W_u, -1), 2
    )
    b_r_u = (
        np.sum(np.maximum(0, w_u_[:, 0]) * np.expand_dims(b_u, -1), 1)
        + np.sum(np.minimum(0, w_u_[:, 0]) * np.expand_dims(b_l, -1), 1)
        + b_u_
    )
    b_r_l = (
        np.sum(np.maximum(0, w_l_[:, 0]) * np.expand_dims(b_l, -1), 1)
        + np.sum(np.minimum(0, w_l_[:, 0]) * np.expand_dims(b_u, -1), 1)
        + b_l_
    )
    if floatx==16:
        output_ref_= None
    # check bounds consistency
    helpers.assert_output_properties_box_linear(x, output_ref_, z[:, 0], z[:, 1], None, w_r_u, b_r_u, None, w_r_l, b_r_l)
