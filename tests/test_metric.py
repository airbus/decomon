# Test unit for decomon with Dense layers


import pytest
import tensorflow.python.keras.backend as K

from decomon.metrics.utils import categorical_cross_entropy


def test_categorical_cross_entropy(odd, mode, floatx, helpers):

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    if floatx == 16:
        K.set_epsilon(1e-2)
    if floatx == 16:
        decimal = 2
    else:
        decimal = 5

    inputs_0 = helpers.get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = helpers.get_standard_values_multid_box(odd, dc_decomp=False)

    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    if mode == "hybrid":
        output = categorical_cross_entropy(inputs_0[2:], dc_decomp=False, mode=mode)
    if mode == "forward":
        output = categorical_cross_entropy([z_0, W_u_0, b_u_0, W_l_0, b_l_0], dc_decomp=False, mode=mode)
    if mode == "ibp":
        output = categorical_cross_entropy([u_c_0, l_c_0], dc_decomp=False, mode=mode)

    f_ref = K.function(inputs_0, -y_0 + K.log(K.sum(K.exp(y_0), -1))[:, None])
    f_entropy = K.function(inputs_0, output)

    y_ = f_ref(inputs_)
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x,
            y_,
            None,
            None,
            z_[:, 0],
            z_[:, 1],
            None,
            w_u_,
            b_u_,
            None,
            w_l_,
            b_l_,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )
    if mode == "ibp":
        u_c_, l_c_ = f_entropy(inputs_)
        helpers.assert_output_properties_box(
            x,
            y_,
            None,
            None,
            inputs_[2][:, 0],
            inputs_[2][:, 1],
            u_c_,
            None,
            None,
            l_c_,
            None,
            None,
            "minus_multid_{}".format(odd),
            decimal=decimal,
        )

    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
