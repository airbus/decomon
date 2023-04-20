# creating toy network and assess that the decomposition is correct


import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential

from decomon.layers.core import ForwardMode
from decomon.models import clone
from decomon.models.convert import ConvertMethod
from decomon.utils import Slope


def test_convert_1D(n, method, mode, floatx, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        return

    K.set_floatx("float{}".format(floatx))
    eps = K.epsilon()
    decimal = 5
    if floatx == 16:
        K.set_epsilon(1e-2)
        decimal = 2

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    ibp = True
    affine = True
    mode = ForwardMode(mode)
    if mode == ForwardMode.AFFINE:
        ibp = False
    if mode == ForwardMode.IBP:
        affine = False

    f_dense = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)

    f_ref = K.function(inputs, ref_nn(inputs[1]))
    y_ref = f_ref(inputs_)

    u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = [None] * 6
    if mode == ForwardMode.HYBRID:
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = f_dense(z_)
    elif mode == ForwardMode.IBP:
        u_c_, l_c_ = f_dense(z_)
    elif mode == ForwardMode.AFFINE:
        z_, w_u_, b_u_, w_l_, b_l_ = f_dense(z_)
    else:
        raise ValueError("Unknown mode.")

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


def test_convert_1D_forward_slope(slope, helpers):
    n, method, mode = 0, "forward-hybrid", "hybrid"
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    ibp = True
    affine = True

    f_dense = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of activation layers
    for layer in f_dense.layers:
        if layer.__class__.__name__.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_convert_1D_backward_slope(slope, helpers):
    n, method, mode = 0, "crown-forward-hybrid", "hybrid"
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    x_ = inputs_[0]
    z_ = inputs_[2]

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(inputs[1])

    ibp = True
    affine = True

    f_dense = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of layers with activation
    for layer in f_dense.layers:
        layer_class_name = layer.__class__.__name__
        if layer_class_name.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_name_forward():
    layers = []
    layers.append(Dense(1, input_dim=1))
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
    layers.append(Dense(1, input_dim=1))
    layers.append(Dense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(Dense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_b = clone(model=model, method=ConvertMethod.CROWN_FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 2
    nb_batman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 3
