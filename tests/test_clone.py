# creating toy network and assess that the decomposition is correct


import pytest
import tensorflow.python.keras.backend as K

from decomon.models import clone


def test_convert_1D(n, method, mode, floatx, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=forward/hybrid
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    IBP = True
    forward = True
    if mode == "forward":
        IBP = False
    if mode == "ibp":
        forward = False

    f_dense = clone(ref_nn, method=method, final_ibp=IBP, final_forward=forward)

    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == "hybrid":
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(z_)
    if mode == "ibp":
        u_c_, l_c_ = f_dense(z_)
    if mode == "forward":
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(z_)

    helpers.assert_output_properties_box(
        x_,
        y_ref,
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
        "dense_{}".format(n),
        decimal=decimal,
    )
    K.set_floatx("float{}".format(32))
    K.set_epsilon(eps)
