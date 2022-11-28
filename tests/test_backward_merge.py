# Test unit for decomon with Dense layers


"""
@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
def test_BackwardAdd_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonAdd(dc_decomp=False)

    backward_op = BackwardAdd(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 + y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 + u_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 + l_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 + y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 + u_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 + l_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
    )


@pytest.mark.parametrize("n_0", [0])
def test_BackwardAdd_multiD_box(n_0):

    inputs_0 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    n_out = y_0.shape[-1]
    w_out = Input((1, n_out, n_out))
    b_out = Input((1, n_out))

    decomon_op = DecomonAdd(dc_decomp=False)

    backward_op = BackwardAdd(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(
        inputs_0_
        + inputs_1_
        + [np.concatenate([np.diag([1.0] * n_out)[None]] * len(x_0))[:, None], np.zeros((len(x_0), 1, n_out))]
    )

    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(
        np.maximum(w_u_0_, 0.0) * np.expand_dims(W_u_0, -1) + np.minimum(w_u_0_, 0.0) * np.expand_dims(W_l_0, -1), 2
    )
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )

    w_l_b_0 = np.sum(
        np.maximum(w_l_0_, 0.0) * np.expand_dims(W_l_0, -1) + np.minimum(w_l_0_, 0.0) * np.expand_dims(W_u_0, -1), 2
    )
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    # w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    w_u_b_1 = np.sum(
        np.maximum(w_u_1_, 0.0) * np.expand_dims(W_u_1, -1) + np.minimum(w_u_1_, 0.0) * np.expand_dims(W_l_1, -1), 2
    )
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    # w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    w_l_b_1 = np.sum(
        np.maximum(w_l_1_, 0.0) * np.expand_dims(W_l_1, -1) + np.minimum(w_l_1_, 0.0) * np.expand_dims(W_u_1, -1), 2
    )
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 + y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 + u_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 + l_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 + y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 + u_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 + l_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_0),
    )


#### Average
@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
def test_BackwardAverage_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonAverage(dc_decomp=False)

    backward_op = BackwardAverage(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        0.5 * (y_0 + y_1),
        z_0[:, 0],
        z_0[:, 1],
        0.5 * (u_c_0 + u_c_1),
        w_u_b_0,
        b_u_b_0,
        0.5 * (l_c_0 + l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        0.5 * (y_0 + y_1),
        z_1[:, 0],
        z_1[:, 1],
        0.5 * (u_c_0 + u_c_1),
        w_u_b_1,
        b_u_b_1,
        0.5 * (l_c_0 + l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
    )


@pytest.mark.parametrize("n_0", [0])
def test_BackwardAdd_multiD_box(n_0):

    inputs_0 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    n_out = y_0.shape[-1]
    w_out = Input((1, n_out, n_out))
    b_out = Input((1, n_out))

    decomon_op = DecomonAverage(dc_decomp=False)

    backward_op = BackwardAverage(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(
        inputs_0_
        + inputs_1_
        + [np.concatenate([np.diag([1.0] * n_out)[None]] * len(x_0))[:, None], np.zeros((len(x_0), 1, n_out))]
    )

    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(
        np.maximum(w_u_0_, 0.0) * np.expand_dims(W_u_0, -1) + np.minimum(w_u_0_, 0.0) * np.expand_dims(W_l_0, -1), 2
    )
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )

    w_l_b_0 = np.sum(
        np.maximum(w_l_0_, 0.0) * np.expand_dims(W_l_0, -1) + np.minimum(w_l_0_, 0.0) * np.expand_dims(W_u_0, -1), 2
    )
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(
        np.maximum(w_u_1_, 0.0) * np.expand_dims(W_u_1, -1) + np.minimum(w_u_1_, 0.0) * np.expand_dims(W_l_1, -1), 2
    )
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(
        np.maximum(w_l_1_, 0.0) * np.expand_dims(W_l_1, -1) + np.minimum(w_l_1_, 0.0) * np.expand_dims(W_u_1, -1), 2
    )
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        0.5 * (y_0 + y_1),
        z_0[:, 0],
        z_0[:, 1],
        0.5 * (u_c_0 + u_c_1),
        w_u_b_0,
        b_u_b_0,
        0.5 * (l_c_0 + l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        0.5 * (y_0 + y_1),
        z_1[:, 0],
        z_1[:, 1],
        0.5 * (u_c_0 + u_c_1),
        w_u_b_1,
        b_u_b_1,
        0.5 * (l_c_0 + l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_0),
    )


## Subtract

#### Average
@pytest.mark.parametrize("n_0, n_1", [(0, 3), (1, 4), (2, 5)])
def test_BackwardSubtract_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonSubtract(dc_decomp=False)

    backward_op = BackwardSubtract(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 - y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 - l_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 - u_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 - y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 - l_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 - u_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
    )


@pytest.mark.parametrize("n_0", [0])
def test_BackwardSubtract_multiD_box(n_0):

    inputs_0 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_0_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_multid_box(n_0, dc_decomp=False)
    inputs_1_ = get_standard_values_multid_box(n_0, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    n_out = y_0.shape[-1]
    w_out = Input((1, n_out, n_out))
    b_out = Input((1, n_out))

    decomon_op = DecomonSubtract(dc_decomp=False)

    backward_op = BackwardSubtract(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)

    output_ = f_add(
        inputs_0_
        + inputs_1_
        + [np.concatenate([np.diag([1.0] * n_out)[None]] * len(x_0))[:, None], np.zeros((len(x_0), 1, n_out))]
    )

    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(
        np.maximum(w_u_0_, 0.0) * np.expand_dims(W_u_0, -1) + np.minimum(w_u_0_, 0.0) * np.expand_dims(W_l_0, -1), 2
    )
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )

    w_l_b_0 = np.sum(
        np.maximum(w_l_0_, 0.0) * np.expand_dims(W_l_0, -1) + np.minimum(w_l_0_, 0.0) * np.expand_dims(W_u_0, -1), 2
    )
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(
        np.maximum(w_u_1_, 0.0) * np.expand_dims(W_u_1, -1) + np.minimum(w_u_1_, 0.0) * np.expand_dims(W_l_1, -1), 2
    )
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(
        np.maximum(w_l_1_, 0.0) * np.expand_dims(W_l_1, -1) + np.minimum(w_l_1_, 0.0) * np.expand_dims(W_u_1, -1), 2
    )
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        y_0 - y_1,
        z_0[:, 0],
        z_0[:, 1],
        u_c_0 - l_c_1,
        w_u_b_0,
        b_u_b_0,
        l_c_0 - u_c_1,
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        y_0 - y_1,
        z_1[:, 0],
        z_1[:, 1],
        u_c_0 - l_c_1,
        w_u_b_1,
        b_u_b_1,
        l_c_0 - u_c_1,
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_0),
    )


#### Maximum
@pytest.mark.parametrize("n_0, n_1", [(0, 3), (2, 2), (1, 4), (2, 5)])
def test_BackwardMaximum_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonMaximum(dc_decomp=False)

    backward_op = BackwardMaximum(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    f_op = K.function(inputs_0 + inputs_1, decomon_op(inputs_0[1:] + inputs_1[1:]))

    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        np.maximum(y_0, y_1),
        z_0[:, 0],
        z_0[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_0,
        b_u_b_0,
        np.maximum(l_c_0, l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        np.maximum(y_0, y_1),
        z_1[:, 0],
        z_1[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_1,
        b_u_b_1,
        np.maximum(l_c_0, l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
    )


#### Maximum
@pytest.mark.parametrize("n_0, n_1", [(0, 0), (0, 3), (1, 4), (2, 5)])
def test_BackwardMinimum_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonMinimum(dc_decomp=False)

    backward_op = BackwardMinimum(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    f_op = K.function(inputs_0 + inputs_1, decomon_op(inputs_0[1:] + inputs_1[1:]))

    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    w_u_b_0 = np.sum(np.maximum(w_u_0_[:, 0], 0) * W_u_0 + np.minimum(w_u_0_[:, 0], 0) * W_l_0, 1)[:, :, None]
    b_u_b_0 = (
        b_u_0_[:, 0]
        + np.sum(np.maximum(w_u_0_[:, 0], 0) * b_u_0[:, :, None], 1)
        + np.sum(np.minimum(w_u_0_[:, 0], 0) * b_l_0[:, :, None], 1)
    )
    w_l_b_0 = np.sum(np.maximum(w_l_0_[:, 0], 0) * W_l_0 + np.minimum(w_l_0_[:, 0], 0) * W_u_0, 1)[:, :, None]
    b_l_b_0 = (
        b_l_0_[:, 0]
        + np.sum(np.maximum(w_l_0_[:, 0], 0) * b_l_0[:, :, None], 1)
        + np.sum(np.minimum(w_l_0_[:, 0], 0) * b_u_0[:, :, None], 1)
    )

    w_u_b_1 = np.sum(np.maximum(w_u_1_[:, 0], 0) * W_u_1 + np.minimum(w_u_1_[:, 0], 0) * W_l_1, 1)[:, :, None]
    b_u_b_1 = (
        b_u_1_[:, 0]
        + np.sum(np.maximum(w_u_1_[:, 0], 0) * b_u_1[:, :, None], 1)
        + np.sum(np.minimum(w_u_1_[:, 0], 0) * b_l_1[:, :, None], 1)
    )
    w_l_b_1 = np.sum(np.maximum(w_l_1_[:, 0], 0) * W_l_1 + np.minimum(w_l_1_[:, 0], 0) * W_u_1, 1)[:, :, None]
    b_l_b_1 = (
        b_l_1_[:, 0]
        + np.sum(np.maximum(w_l_1_[:, 0], 0) * b_l_1[:, :, None], 1)
        + np.sum(np.minimum(w_l_1_[:, 0], 0) * b_u_1[:, :, None], 1)
    )

    assert_output_properties_box_linear(
        x_0,
        np.maximum(y_0, y_1),
        z_0[:, 0],
        z_0[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_0,
        b_u_b_0,
        np.maximum(l_c_0, l_c_1),
        w_l_b_0,
        b_l_b_0,
        "dense_{}".format(n_0),
    )

    assert_output_properties_box_linear(
        x_1,
        np.maximum(y_0, y_1),
        z_1[:, 0],
        z_1[:, 1],
        np.maximum(u_c_0, u_c_1),
        w_u_b_1,
        b_u_b_1,
        np.maximum(l_c_0, l_c_1),
        w_l_b_1,
        b_l_b_1,
        "dense_{}".format(n_1),
    )


@pytest.mark.parametrize("n_0, n_1", [(0, 0), (0, 3), (1, 4), (2, 5)])
def test_BackwardMultiply_1D_box(n_0, n_1):

    inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
    x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

    inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
    inputs_1_ = get_standart_values_1d_box(n_1, dc_decomp=False)
    x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

    w_out = Input((1, 1, 1))
    b_out = Input((1, 1))

    decomon_op = DecomonMultiply(dc_decomp=False)

    backward_op = BackwardMultiply(layer=decomon_op)

    back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])
    f_decomon = K.function(inputs_0 + inputs_1, decomon_op(inputs_0[1:] + inputs_1[1:]))

    f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
    output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
    y_, _, u_, _, _, l_, _, _ = f_decomon(inputs_0_ + inputs_1_)

    w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

    assert_output_properties_box_linear(
        y_0,
        y_,
        l_c_0,
        u_c_0,
        u_,
        w_u_0_[:, 0],
        b_u_0_[:, 0],
        l_,
        w_l_0_[:, 0],
        b_l_0_[:, 0],
        "dense_{}".format(n_0),
    )


def test_BackwardDot_1D_box(n_0=0):

    try:

        inputs_0 = get_tensor_decomposition_1d_box(dc_decomp=False)
        inputs_0_ = get_standart_values_1d_box(n_0, dc_decomp=False)
        x_0, y_0, z_0, u_c_0, W_u_0, b_u_0, l_c_0, W_l_0, b_l_0 = inputs_0_

        inputs_1 = get_tensor_decomposition_1d_box(dc_decomp=False)
        inputs_1_ = get_standart_values_1d_box(n_0, dc_decomp=False)
        x_1, y_1, z_1, u_c_1, W_u_1, b_u_1, l_c_1, W_l_1, b_l_1 = inputs_1_

        w_out = Input((1, 1, 1))
        b_out = Input((1, 1))

        decomon_op = DecomonDot(dc_decomp=False)

        backward_op = BackwardDot(layer=decomon_op)

        back_bounds_0, back_bounds_1 = backward_op(inputs_0[1:] + inputs_1[1:] + [w_out, b_out, w_out, b_out])

        f_decomon = K.function(inputs_0 + inputs_1, decomon_op(inputs_0[1:] + inputs_1[1:]))

        f_add = K.function(inputs_0 + inputs_1 + [w_out, b_out], back_bounds_0 + back_bounds_1)
        output_ = f_add(inputs_0_ + inputs_1_ + [np.ones((len(x_0), 1, 1, 1)), np.zeros((len(x_0), 1, 1))])
        y_, _, u_, _, _, l_, _, _ = f_decomon(inputs_0_ + inputs_1_)

        w_u_0_, b_u_0_, w_l_0_, b_l_0_, w_u_1_, b_u_1_, w_l_1_, b_l_1_ = output_

        assert_output_properties_box_linear(
            y_0,
            y_,
            l_c_0,
            u_c_0,
            u_,
            w_u_0_[:, 0],
            b_u_0_[:, 0],
            l_,
            w_l_0_[:, 0],
            b_l_0_[:, 0],
            "dense_{}".format(n_0),
            decimal=4,
        )

    except NotImplementedError:
        print("kikou")
"""
