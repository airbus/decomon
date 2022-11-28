"""
@pytest.mark.parametrize(
    "n, n_subgrad, slope",
    [
        (0, 0, "volume-slope"),
        (1, 0, "volume-slope"),
        (2, 0, "volume-slope"),
        (3, 0, "volume-slope"),
        (4, 0, "volume-slope"),
        (5, 0, "volume-slope"),
        (0, 1, "volume-slope"),
        (1, 1, "volume-slope"),
        (2, 1, "volume-slope"),
        (3, 1, "volume-slope"),
        (4, 1, "volume-slope"),
        (5, 1, "volume-slope"),
        (0, 5, "volume-slope"),
        (1, 5, "volume-slope"),
        (2, 5, "volume-slope"),
        (3, 5, "volume-slope"),
        (4, 5, "volume-slope"),
        (5, 5, "volume-slope"),
        (0, 0, "same-slope"),
        (1, 0, "same-slope"),
        (2, 0, "same-slope"),
        (3, 0, "same-slope"),
        (4, 0, "same-slope"),
        (5, 0, "same-slope"),
        (0, 1, "same-slope"),
        (1, 1, "same-slope"),
        (2, 1, "same-slope"),
        (3, 1, "same-slope"),
        (4, 1, "same-slope"),
        (5, 1, "same-slope"),
        (0, 5, "same-slope"),
        (1, 5, "same-slope"),
        (2, 5, "same-slope"),
        (3, 5, "same-slope"),
        (4, 5, "same-slope"),
        (5, 5, "same-slope"),
    ],
)
def test_convert_backward_model_1d_box_nodc(n, n_subgrad, slope):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="linear", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential, mode='forward', dc_decomp=False)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_


    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    backward_model = get_backward(monotonic_model, slope=slope)

    import pdb; pdb.set_trace()
    output = backward_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)

    # test that nothing has changed with the forward mode
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, "nodc")

    # test that the backward mode is correct
    assert_output_properties_box_linear(
        x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_b[:, 0], b_u_b[:, 0], l_c_, w_l_b[:, 0], b_l_b[:, 0], "nodc"
    )
"""


"""
@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_backward_model_1d_box(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = convert(sequential, dc_decomp=False, mode="forward", forward=True)
    backward_model = get_backward(monotonic_model)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, _, _, _, _, _, _ = inputs
    x_i, y_i, z_i, _, _, _, _, _, _ = inputs_

    output_ref = sequential(inputs[0])
    f_ref = K.function(inputs, output_ref)

    output = backward_model([y, z])

    f_clone = K.function([y, z], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone([y_i, z_i])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(
        x_i,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_f,
        b_u_f,
        l_c_,
        w_l_f,
        b_l_f,
        "clone_sequential_{}".format(n),
        decimal=5,
    )

    assert_output_properties_box_linear(
        x_i,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_b[:, 0],
        b_u_b[:, 0],
        l_c_,
        w_l_b[:, 0],
        b_l_b[:, 0],
        "clone_sequential_{}".format(n),
        decimal=5,
    )


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_clone_backward_sequential_model_multid_box(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    monotonic_model = clone_sequential_model(sequential, input_dim=input_dim, dc_decomp=False)
    backward_model = get_backward(monotonic_model)

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = backward_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=6, err_msg="reconstruction error")
    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_f,
        b_u_f,
        l_c_,
        w_l_f,
        b_l_f,
        "clone_sequential_{}".format(odd),
        decimal=5,
    )

    w_u_ = np.sum(np.maximum(w_u_b, 0) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(w_u_b, 0) * np.expand_dims(W_l, -1), 2
    )
    b_u_ = (
        np.sum(np.maximum(w_u_b, 0) * np.expand_dims(np.expand_dims(b_u, -1), -1), (1, 2))
        + np.sum(np.minimum(w_u_b, 0) * np.expand_dims(np.expand_dims(b_l, -1), -1), (1, 2))
        + b_u_b[:, 0]
    )

    w_l_ = np.sum(np.maximum(w_l_b, 0) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(w_l_b, 0) * np.expand_dims(W_u, -1), 2
    )
    b_l_ = (
        np.sum(np.maximum(w_l_b, 0) * np.expand_dims(np.expand_dims(b_l, -1), -1), (1, 2))
        + np.sum(np.minimum(w_l_b, 0) * np.expand_dims(np.expand_dims(b_u, -1), -1), (1, 2))
        + b_l_b[:, 0]
    )

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "clone_sequential_{}".format(odd),
        decimal=5,
    )


# BACKWARD mode integrated in the conversion


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_bacwkard_model_1d_box_nodc_mode(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    backward_model = clone(sequential, input_dim=1, dc_decomp=False, mode="backward")

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)

    output = backward_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)

    # test that nothing has changed with the forward mode
    assert_output_properties_box_linear(x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, "nodc")

    # test that the backward mode is correct
    assert_output_properties_box_linear(
        x, y_, z_[:, 0], z_[:, 1], u_c_, w_u_b[:, 0], b_u_b[:, 0], l_c_, w_l_b[:, 0], b_l_b[:, 0], "nodc"
    )


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_backward_model_1d_box_mode(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    backward_model = convert(sequential, dc_decomp=False, mode="backward", forward=True)

    inputs = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)
    x, _, z, _, _, _, _, _, _ = inputs
    x_i, _, z_i, _, _, _, _, _, _ = inputs_

    output_ref = sequential(inputs[0])
    f_ref = K.function(inputs, output_ref)

    output = backward_model([x, z])

    f_clone = K.function([x, z], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone([x_i, z_i])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=5)
    assert_output_properties_box_linear(
        x_i,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_f,
        b_u_f,
        l_c_,
        w_l_f,
        b_l_f,
        "clone_sequential_{}".format(n),
        decimal=5,
    )

    assert_output_properties_box_linear(
        x_i,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_b[:, 0],
        b_u_b[:, 0],
        l_c_,
        w_l_b[:, 0],
        b_l_b[:, 0],
        "clone_sequential_{}".format(n),
        decimal=5,
    )


@pytest.mark.parametrize("odd, n_subgrad", [(0, 0), (1, 0), (0, 1), (1, 1), (0, 5), (1, 5)])
def test_clone_backward_sequential_model_multid_box_mode(odd, n_subgrad):

    inputs = get_tensor_decomposition_multid_box(odd, dc_decomp=False)
    inputs_ = get_standard_values_multid_box(odd, dc_decomp=False)
    x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs_

    input_dim = x.shape[-1]

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(1, activation="relu", input_dim=y.shape[-1]))
    sequential.add(Dense(1, activation="linear"))
    backward_model = clone(sequential, input_dim=input_dim, dc_decomp=False, mode="backward")

    output_ref = sequential(inputs[1])
    f_ref = K.function(inputs, output_ref)
    output = backward_model(inputs[1:])

    f_clone = K.function(inputs[1:], output)
    y_, z_, u_c_, w_u_f, b_u_f, l_c_, w_l_f, b_l_f, w_u_b, b_u_b, w_l_b, b_l_b = f_clone(inputs_[1:])
    y_ref = f_ref(inputs_)

    assert_almost_equal(y_, y_ref, decimal=6, err_msg="reconstruction error")
    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_f,
        b_u_f,
        l_c_,
        w_l_f,
        b_l_f,
        "clone_sequential_{}".format(odd),
        decimal=5,
    )

    w_u_ = np.sum(np.maximum(w_u_b, 0) * np.expand_dims(W_u, -1), 2) + np.sum(
        np.minimum(w_u_b, 0) * np.expand_dims(W_l, -1), 2
    )
    b_u_ = (
        np.sum(np.maximum(w_u_b, 0) * np.expand_dims(np.expand_dims(b_u, -1), -1), (1, 2))
        + np.sum(np.minimum(w_u_b, 0) * np.expand_dims(np.expand_dims(b_l, -1), -1), (1, 2))
        + b_u_b[:, 0]
    )

    w_l_ = np.sum(np.maximum(w_l_b, 0) * np.expand_dims(W_l, -1), 2) + np.sum(
        np.minimum(w_l_b, 0) * np.expand_dims(W_u, -1), 2
    )
    b_l_ = (
        np.sum(np.maximum(w_l_b, 0) * np.expand_dims(np.expand_dims(b_l, -1), -1), (1, 2))
        + np.sum(np.minimum(w_l_b, 0) * np.expand_dims(np.expand_dims(b_u, -1), -1), (1, 2))
        + b_l_b[:, 0]
    )

    assert_output_properties_box_linear(
        x,
        y_,
        z_[:, 0],
        z_[:, 1],
        u_c_,
        w_u_,
        b_u_,
        l_c_,
        w_l_,
        b_l_,
        "clone_sequential_{}".format(odd),
        decimal=5,
    )


# Backward does not modify forward


@pytest.mark.parametrize(
    "n, n_subgrad",
    [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
    ],
)
def test_convert_bacwkard_model_1d_box_nodc_forward(n, n_subgrad):

    # build a simple sequential model from keras
    # start with 1D
    sequential = Sequential()
    sequential.add(Dense(10, activation="relu", input_dim=1))
    sequential.add(Dense(1, activation="linear"))

    monotonic_model = clone(sequential, input_dim=1, dc_decomp=False)
    inputs_ = get_standart_values_1d_box(n, dc_decomp=False)

    backward_model = get_backward(monotonic_model)

    _, _, u_c_f, w_u_f, b_u_f, l_c_f, w_l_f, b_l_f = monotonic_model.predict(inputs_[1:])
    _, _, u_c_b, w_u_b, b_u_b, l_c_b, w_l_b, b_l_b, _, _, _, _ = backward_model.predict(inputs_[1:])

    assert_almost_equal(u_c_f, u_c_b, decimal=5, err_msg="upper")
    assert_almost_equal(l_c_f, l_c_b, decimal=5, err_msg="lower")
    assert_almost_equal(w_u_f, w_u_b, decimal=5, err_msg="w_u")
    assert_almost_equal(w_l_f, w_l_b, decimal=5, err_msg="w_l")
    assert_almost_equal(b_u_f, b_u_b, decimal=5, err_msg="b_u")
    assert_almost_equal(b_l_f, b_l_b, decimal=5, err_msg="b_l")
"""
