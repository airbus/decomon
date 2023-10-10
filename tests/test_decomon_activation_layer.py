from decomon.core import ForwardMode, Slope
from decomon.layers.decomon_layers import DecomonActivation


def test_decomon_activation_slope(helpers):
    mode = ForwardMode.AFFINE
    activation = "relu"
    n = 2

    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=False)
    (
        x_0,
        y_0,
        z_0,
        u_c_0,
        W_u_0,
        b_u_0,
        l_c_0,
        W_l_0,
        b_l_0,
    ) = inputs_  # numpy values

    outputs_by_slope = {}
    for slope in Slope:
        layer = DecomonActivation(activation, dc_decomp=False, mode=mode, slope=slope)
        assert layer.slope == slope
        inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=False)
        x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs  # tensors
        inputs_for_mode = [z, W_u, b_u, W_l, b_l]
        output = layer(inputs_for_mode)
        f_func = helpers.function(inputs, output)
        outputs_by_slope[slope] = f_func(inputs_)

    # check results
    # O_Slope != Z_Slope
    same_outputs_O_n_Z = [
        (a == b).all() for a, b in zip(outputs_by_slope[Slope.O_SLOPE], outputs_by_slope[Slope.Z_SLOPE])
    ]
    assert not all(same_outputs_O_n_Z)

    # V_Slope == Z_Slope
    for a, b in zip(outputs_by_slope[Slope.V_SLOPE], outputs_by_slope[Slope.Z_SLOPE]):
        assert (a == b).all()
