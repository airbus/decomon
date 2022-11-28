"""
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
"""
