from __future__ import absolute_import
import numpy as np
from tensorflow.keras import Input
from numpy.testing import assert_almost_equal


def get_standart_values_1d_box(n, dc_decomp=True, grad_bounds=False, nb=100):
    """

    :param n: A set of functions with their monotonic
    decomposition for testing the activations
    :return:
    """
    """
    A set of functions with their monotonic decomposition
    for testing the activations
    """

    w_u_ = np.ones(nb)
    b_u_ = np.zeros(nb)
    w_l_ = np.ones(nb)
    b_l_ = np.zeros(nb)

    if n == 0:
        # identity
        y_ = np.linspace(-2, -1, nb)
        x_ = np.linspace(-2, -1, nb)
        h_ = np.linspace(-2, -1, nb)
        g_ = np.zeros_like(x_)

    if n == 1:
        y_ = np.linspace(1, 2, nb)
        x_ = np.linspace(1, 2, nb)
        h_ = np.linspace(1, 2, nb)
        g_ = np.zeros_like(x_)

    if n == 2:
        y_ = np.linspace(-1, 1, nb)
        x_ = np.linspace(-1, 1, nb)
        h_ = np.linspace(-1, 1, nb)
        g_ = np.zeros_like(x_)

    if n == 3:
        # identity
        y_ = np.linspace(-2, -1, nb)
        x_ = np.linspace(-2, -1, nb)
        h_ = 2 * np.linspace(-2, -1, nb)
        g_ = -np.linspace(-2, -1, nb)

    if n == 4:
        y_ = np.linspace(1, 2, nb)
        x_ = np.linspace(1, 2, nb)
        h_ = 2 * np.linspace(1, 2, nb)
        g_ = -np.linspace(1, 2, nb)

    if n == 5:
        y_ = np.linspace(-1, 1, nb)
        x_ = np.linspace(-1, 1, nb)
        h_ = 2 * np.linspace(-1, 1, nb)
        g_ = -np.linspace(-1, 1, nb)

    if n == 6:

        assert nb == 100, "expected nb=100 samples"
        # cosine function
        x_ = np.linspace(-np.pi, np.pi, 100)
        y_ = np.cos(x_)
        h_ = np.concatenate([y_[:50], np.ones((50,))]) - 0.5
        g_ = np.concatenate([np.ones((50,)), y_[50:]]) - 0.5
        w_u_ = np.zeros_like(x_)
        w_l_ = np.zeros_like(x_)
        b_u_ = np.ones_like(x_)
        b_l_ = -np.ones_like(x_)

    if n == 7:
        # h and g >0
        h_ = np.linspace(0.5, 2, nb)
        g_ = np.linspace(1, 2, nb)[::-1]
        x_ = h_ + g_
        y_ = h_ + g_

    if n == 8:
        # h <0 and g <0
        # h_max+g_max <=0
        h_ = np.linspace(-2, -1, nb)
        g_ = np.linspace(-2, -1, nb)[::-1]
        y_ = h_ + g_
        x_ = h_ + g_

    if n == 9:
        # h >0 and g <0
        # h_min+g_min >=0
        h_ = np.linspace(4, 5, nb)
        g_ = np.linspace(-2, -1, nb)[::-1]
        y_ = h_ + g_
        x_ = h_ + g_

    x_min_ = x_.min() + np.zeros_like(x_)
    x_max_ = x_.max() + np.zeros_like(x_)

    x_0_ = np.concatenate([x_min_[:, None], x_max_[:, None]], 1)

    u_c_ = np.max(y_) * np.ones((nb,))
    l_c_ = np.min(y_) * np.ones((nb,))

    if dc_decomp:

        return [
            x_[:, None],
            y_[:, None],
            x_0_[:, :, None],
            u_c_[:, None],
            w_u_[:, None, None],
            b_u_[:, None],
            l_c_[:, None],
            w_l_[:, None, None],
            b_l_[:, None],
            h_[:, None],
            g_[:, None],
        ]

    return [
        x_[:, None],
        y_[:, None],
        x_0_[:, :, None],
        u_c_[:, None],
        w_u_[:, None, None],
        b_u_[:, None],
        l_c_[:, None],
        w_l_[:, None, None],
        b_l_[:, None],
    ]


def get_tensor_decomposition_1d_box(dc_decomp=True):

    if dc_decomp:
        return [
            Input((1,)),
            Input((1,)),
            Input((2, 1)),
            Input((1,)),
            Input((1, 1)),
            Input((1,)),
            Input((1,)),
            Input((1, 1)),
            Input((1,)),
            Input((1,)),
            Input((1,)),
        ]
    return [
        Input((1,)),
        Input((1,)),
        Input((2, 1)),
        Input((1,)),
        Input((1, 1)),
        Input((1,)),
        Input((1,)),
        Input((1, 1)),
        Input((1,)),
    ]


def get_tensor_decomposition_multid_box(odd=1, dc_decomp=True):

    if odd:
        n = 3
    else:
        n = 2

    if dc_decomp:
        # x, y, z, u, w_u, b_u, l, w_l, b_l, h, g
        return [
            Input((n,)),
            Input((n,)),
            Input((2, n)),
            Input((n,)),
            Input((n, n)),
            Input((n,)),
            Input((n,)),
            Input((n, n)),
            Input((n,)),
            Input((n,)),
            Input((n,)),
        ]
    return [
        Input((n,)),
        Input((n,)),
        Input((2, n)),
        Input((n,)),
        Input((n, n)),
        Input((n,)),
        Input((n,)),
        Input((n, n)),
        Input((n,)),
    ]


def get_standard_values_multid_box(odd=1, dc_decomp=True):

    if dc_decomp:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
            h_0,
            g_0,
        ) = get_standart_values_1d_box(0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
            h_1,
            g_1,
        ) = get_standart_values_1d_box(1, dc_decomp)
    else:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
        ) = get_standart_values_1d_box(0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
        ) = get_standart_values_1d_box(1, dc_decomp)

    if not odd:

        # output (x_0+x_1, x_0+2*x_0)
        x_ = np.concatenate([x_0, x_1], -1)
        z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
        z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
        z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
        y_ = np.concatenate([y_0 + y_1, y_0 + 2 * y_1], -1)
        b_u_ = np.concatenate([b_u_0 + b_u_1, b_u_0 + 2 * b_u_1], -1)
        u_c_ = np.concatenate([u_c_0 + u_c_1, u_c_0 + 2 * u_c_1], -1)
        b_l_ = np.concatenate([b_l_0 + b_l_1, b_l_0 + 2 * b_l_1], -1)
        l_c_ = np.concatenate([l_c_0 + l_c_1, l_c_0 + 2 * l_c_1], -1)

        if dc_decomp:
            h_ = np.concatenate([h_0 + h_1, h_0 + 2 * h_1], -1)
            g_ = np.concatenate([g_0 + g_1, g_0 + 2 * g_1], -1)

        w_u_ = np.zeros((len(x_), 2, 2))
        w_u_[:, 0, 0] = w_u_0[:, 0, 0]
        w_u_[:, 1, 0] = w_u_1[:, 0, 0]
        w_u_[:, 0, 1] = w_u_0[:, 0, 0]
        w_u_[:, 1, 1] = 2 * w_u_1[:, 0, 0]

        w_l_ = np.zeros((len(x_), 2, 2))
        w_l_[:, 0, 0] = w_l_0[:, 0, 0]
        w_l_[:, 1, 0] = w_l_1[:, 0, 0]
        w_l_[:, 0, 1] = w_l_0[:, 0, 0]
        w_l_[:, 1, 1] = 2 * w_l_1[:, 0, 0]

    else:

        (
            x_2,
            y_2,
            z_2,
            u_c_2,
            w_u_2,
            b_u_2,
            l_c_2,
            w_l_2,
            b_l_2,
            h_2,
            g_2,
        ) = get_standart_values_1d_box(2)

        # output (x_0+x_1, x_0+2*x_0, x_2)
        x_ = np.concatenate([x_0, x_1, x_2], -1)
        z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0], z_2[:, 0]], -1)
        z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1], z_2[:, 1]], -1)
        z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
        y_ = np.concatenate([y_0 + y_1, y_0 + 2 * y_1, y_2], -1)
        b_u_ = np.concatenate([b_u_0 + b_u_1, b_u_0 + 2 * b_u_1, b_u_2], -1)
        b_l_ = np.concatenate([b_l_0 + b_l_1, b_l_0 + 2 * b_l_1, b_l_2], -1)
        u_c_ = np.concatenate([u_c_0 + u_c_1, u_c_0 + 2 * u_c_1, u_c_2], -1)
        l_c_ = np.concatenate([l_c_0 + l_c_1, l_c_0 + 2 * l_c_1, l_c_2], -1)

        if dc_decomp:
            h_ = np.concatenate([h_0 + h_1, h_0 + 2 * h_1, h_2], -1)
            g_ = np.concatenate([g_0 + g_1, g_0 + 2 * g_1, g_2], -1)

        w_u_ = np.zeros((len(x_), 3, 3))
        w_u_[:, 0, 0] = w_u_0[:, 0, 0]
        w_u_[:, 1, 0] = w_u_1[:, 0, 0]
        w_u_[:, 0, 1] = w_u_0[:, 0, 0]
        w_u_[:, 1, 1] = 2 * w_u_1[:, 0, 0]
        w_u_[:, 2, 2] = w_u_2[:, 0, 0]

        w_l_ = np.zeros((len(x_), 3, 3))
        w_l_[:, 0, 0] = w_l_0[:, 0, 0]
        w_l_[:, 1, 0] = w_l_1[:, 0, 0]
        w_l_[:, 0, 1] = w_l_0[:, 0, 0]
        w_l_[:, 1, 1] = 2 * w_l_1[:, 0, 0]
        w_l_[:, 2, 2] = w_l_2[:, 0, 0]

    if dc_decomp:
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def build_image_from_1D_box(odd=0, m=0, dc_decomp=True):

    if odd:
        n = 7
    else:
        n = 6

    if dc_decomp:
        (
            x_,
            y_0,
            z_,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
            h_0,
            g_0,
        ) = get_standart_values_1d_box(m, dc_decomp=dc_decomp)
    else:
        (
            x_,
            y_0,
            z_,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
        ) = get_standart_values_1d_box(m, dc_decomp=dc_decomp)

    y_ = np.concatenate([(i + 1) * y_0 for i in range(n * n)], -1).reshape((-1, n, n))
    b_u_ = np.concatenate([(i + 1) * b_u_0 for i in range(n * n)], -1).reshape((-1, n, n))
    b_l_ = np.concatenate([(i + 1) * b_l_0 for i in range(n * n)], -1).reshape((-1, n, n))

    if dc_decomp:
        h_ = np.concatenate([(i + 1) * h_0 for i in range(n * n)], -1).reshape((-1, n, n))
        g_ = np.concatenate([(i + 1) * g_0 for i in range(n * n)], -1).reshape((-1, n, n))

    u_c_ = np.concatenate([(i + 1) * u_c_0 for i in range(n * n)], -1).reshape((-1, n, n))
    l_c_ = np.concatenate([(i + 1) * l_c_0 for i in range(n * n)], -1).reshape((-1, n, n))

    w_u_ = np.zeros((len(x_), 1, n * n))
    w_l_ = np.zeros((len(x_), 1, n * n))

    for i in range(n * n):
        w_u_[:, 0, i] = (i + 1) * w_u_0[:, 0, 0]
        w_l_[:, 0, i] = (i + 1) * w_l_0[:, 0, 0]

    w_u_ = w_u_.reshape((-1, 1, n, n))
    w_l_ = w_l_.reshape((-1, 1, n, n))

    if dc_decomp:
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def build_image_from_2D_box(odd=0, m0=0, m1=1, dc_decomp=True):

    if dc_decomp:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
            h_0,
            g_0,
        ) = build_image_from_1D_box(odd, m0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
            h_1,
            g_1,
        ) = build_image_from_1D_box(odd, m1, dc_decomp)
    else:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
        ) = build_image_from_1D_box(odd, m0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
        ) = build_image_from_1D_box(odd, m1, dc_decomp)

    x_ = np.concatenate([x_0, x_1], -1)
    z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
    z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
    z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
    y_ = y_0 + y_1
    b_u_ = b_u_0 + b_u_1
    b_l_ = b_l_0 + b_l_1

    u_c_ = u_c_0 + u_c_1
    l_c_ = l_c_0 + l_c_1

    w_u_ = np.concatenate([w_u_0, w_u_1], 1)
    w_l_ = np.concatenate([w_l_0, w_l_1], 1)

    if dc_decomp:
        h_ = h_0 + h_1
        g_ = g_0 + g_1

        assert_output_properties_box(
            x_,
            h_ + g_,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            "image from 2D",
        )

        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def get_standard_values_images_box(data_format="channels_last", odd=0, m0=0, m1=1, dc_decomp=True):

    output = build_image_from_2D_box(odd, m0, m1, dc_decomp)
    if dc_decomp:
        x_0, y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = output
    else:
        x_0, y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = output

    x_ = x_0
    z_ = z_0
    z_min_ = z_0[:, 0]
    z_max_ = z_0[:, 1]

    if data_format == "channels_last":

        y_0 = y_0[:, :, :, None]
        b_u_0 = b_u_0[:, :, :, None]
        b_l_0 = b_l_0[:, :, :, None]
        u_c_0 = u_c_0[:, :, :, None]
        l_c_0 = l_c_0[:, :, :, None]
        w_u_0 = w_u_0[:, :, :, :, None]
        w_l_0 = w_l_0[:, :, :, :, None]
        y_ = np.concatenate([y_0, y_0], -1)
        b_u_ = np.concatenate([b_u_0, b_u_0], -1)
        b_l_ = np.concatenate([b_l_0, b_l_0], -1)
        u_c_ = np.concatenate([u_c_0, u_c_0], -1)
        l_c_ = np.concatenate([l_c_0, l_c_0], -1)
        w_u_ = np.concatenate([w_u_0, w_u_0], -1)
        w_l_ = np.concatenate([w_l_0, w_l_0], -1)

        if dc_decomp:
            h_0 = h_0[:, :, :, None]
            g_0 = g_0[:, :, :, None]
            h_ = np.concatenate([h_0, h_0], -1)
            g_ = np.concatenate([g_0, g_0], -1)

            assert_output_properties_box(
                x_,
                y_,
                h_,
                g_,
                z_min_,
                z_max_,
                u_c_,
                w_u_,
                b_u_,
                l_c_,
                w_l_,
                b_l_,
                "images {},{},{},{}".format(data_format, odd, m0, m1),
            )

    else:
        y_0 = y_0[:, None, :, :]
        b_u_0 = b_u_0[:, None, :, :]
        b_l_0 = b_l_0[:, None, :, :]
        u_c_0 = u_c_0[:, None, :, :]
        l_c_0 = l_c_0[:, None, :, :]
        w_u_0 = w_u_0[:, :, None, :, :]
        w_l_0 = w_l_0[:, :, None, :, :]

        y_ = np.concatenate([y_0, y_0], 1)
        b_u_ = np.concatenate([b_u_0, b_u_0], 2)
        b_l_ = np.concatenate([b_l_0, b_l_0], 2)
        u_c_ = np.concatenate([u_c_0, u_c_0], 2)
        l_c_ = np.concatenate([l_c_0, l_c_0], 2)
        w_u_ = np.concatenate([w_u_0, w_u_0], 2)
        w_l_ = np.concatenate([w_l_0, w_l_0], 2)

        if dc_decomp:
            h_0 = h_0[:, None, :, :]
            g_0 = g_0[:, None, :, :]
            h_ = np.concatenate([h_0, h_0], 1)
            g_ = np.concatenate([g_0, g_0], 1)
            assert_output_properties_box(
                x_,
                y_,
                h_,
                g_,
                z_min_,
                z_max_,
                u_c_,
                w_u_,
                b_u_,
                l_c_,
                w_l_,
                b_l_,
                "images {},{},{},{}".format(data_format, odd, m0, m1),
            )

    if dc_decomp:
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    else:
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def get_tensor_decomposition_images_box(data_format, odd, dc_decomp=True):

    if odd:
        n = 7
    else:
        n = 6

    if data_format == "channels_last":
        # x, y, z, u, w_u, b_u, l, w_l, b_l

        output = [
            Input((2,)),
            Input((n, n, 2)),
            Input((2, 2)),
            Input((n, n, 2)),
            Input((2, n, n, 2)),
            Input((n, n, 2)),
            Input((n, n, 2)),
            Input((2, n, n, 2)),
            Input((n, n, 2)),
        ]

        if dc_decomp:
            output += [Input((n, n, 2)), Input((n, n, 2))]
    else:

        output = [
            Input((2,)),
            Input((2, n, n)),
            Input((2, 2)),
            Input((2, n, n)),
            Input((2, 2, n, n)),
            Input((2, n, n)),
            Input((2, n, n)),
            Input((2, 2, n, n)),
            Input((2, n, n)),
        ]
        if dc_decomp:

            output += [Input((n, n, 2)), Input((n, n, 2))]

    return output


def assert_output_properties_box(x_, y_, h_, g_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, name, decimal=5):

    assert_almost_equal(
        h_ + g_,
        y_,
        decimal=decimal,
        err_msg="decomposition error for function {}".format(name),
    )
    assert np.min(x_min_ <= x_max_), "x_min >x_max for function {}".format(name)

    assert_almost_equal(
        np.clip(x_min_ - x_, 0, np.inf),
        0.0,
        decimal=decimal,
        err_msg="x_min >x_  for function {}".format(name),
    )
    assert_almost_equal(
        np.clip(x_ - x_max_, 0, np.inf),
        0.0,
        decimal=decimal,
        err_msg="x_max < x_  for function {}".format(name),
    )

    x_expand = x_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand.shape)
    for i in range(n_expand):
        x_expand = np.expand_dims(x_expand, -1)

    lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
    upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

    # check that the functions h_ and g_ remains monotonic

    assert_almost_equal(
        np.clip(h_[:-1] - h_[1:], 0, np.inf),
        np.zeros_like(h_[1:]),
        decimal=decimal,
        err_msg="h is not increasing for function {}".format(name),
    )
    assert_almost_equal(
        np.clip(g_[1:] - g_[:-1], 0, np.inf),
        np.zeros_like(g_[1:]),
        decimal=decimal,
        err_msg="g is not increasing for function {}".format(name),
    )

    assert_almost_equal(
        np.clip(l_c_ - y_, 0.0, np.inf),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="l_c >y",
    )
    assert_almost_equal(
        np.clip(y_ - u_c_, 0.0, 1e6),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="u_c <y",
    )

    #

    assert_almost_equal(
        np.clip(lower_ - y_, 0.0, np.inf),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="lower_ >y",
    )
    assert_almost_equal(
        np.clip(y_ - upper_, 0.0, 1e6),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="upper <y",
    )

    # computer lower bounds on the domain

    x_expand_min = x_min_ + np.zeros_like(x_)
    x_expand_max = x_max_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand_min.shape)
    for i in range(n_expand):
        x_expand_min = np.expand_dims(x_expand_min, -1)
        x_expand_max = np.expand_dims(x_expand_max, -1)

    lower_ = np.sum(np.maximum(0, w_l_) * x_expand_min, 1) + np.sum(np.minimum(0, w_l_) * x_expand_max, 1) + b_l_
    upper_ = np.sum(np.maximum(0, w_u_) * x_expand_max, 1) + np.sum(np.minimum(0, w_u_) * x_expand_min, 1) + b_u_

    assert_almost_equal(
        np.clip(lower_.min(0) - l_c_.max(0), 0.0, np.inf),
        np.zeros_like(y_.min(0)),
        decimal=decimal,
        err_msg="lower_ >l_c",
    )
    assert_almost_equal(
        np.clip(u_c_.min(0) - upper_.max(0), 0.0, 1e6),
        np.zeros_like(y_.min(0)),
        decimal=decimal,
        err_msg="upper <u_c",
    )


def assert_output_properties_box_linear(x_, y_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, name, decimal=5):

    # flatten everything
    # flatten everyting
    n = len(y_)
    y_ = y_.reshape((n, -1))
    u_c_ = u_c_.reshape((n, -1))
    l_c_ = l_c_.reshape((n, -1))
    w_u_ = w_u_.reshape((n, w_u_.shape[1], -1))
    w_l_ = w_l_.reshape((n, w_l_.shape[1], -1))
    b_u_ = b_u_.reshape((n, -1))
    b_l_ = b_l_.reshape((n, -1))

    # assert_almost_equal(h_ + g_, y_, decimal=decimal, err_msg='decomposition error for function {}'.format(name))
    assert np.min(x_min_ <= x_max_), "x_min >x_max for function {}".format(name)

    assert_almost_equal(
        np.clip(x_min_ - x_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_min >x_  for function {}".format(name)
    )
    assert_almost_equal(
        np.clip(x_ - x_max_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_max < x_  for function {}".format(name)
    )

    x_expand = x_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand.shape)
    for i in range(n_expand):
        x_expand = np.expand_dims(x_expand, -1)

    lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
    upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

    # check that the functions h_ and g_ remains monotonic

    # assert_almost_equal(np.clip(h_[:-1] - h_[1:], 0, np.inf), np.zeros_like(h_[1:]), decimal=decimal,
    #                    err_msg='h is not increasing for function {}'.format(name))
    # assert_almost_equal(np.clip(g_[1:] - g_[:-1], 0, np.inf), np.zeros_like(g_[1:]), decimal=decimal,
    #                    err_msg='g is not increasing for function {}'.format(name))

    assert_almost_equal(np.clip(l_c_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="l_c >y")
    assert_almost_equal(np.clip(y_ - u_c_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="u_c <y")

    #
    # import pdb; pdb.set_trace()
    assert_almost_equal(np.clip(lower_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="lower_ >y")
    assert_almost_equal(np.clip(y_ - upper_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="upper <y")

    # computer lower bounds on the domain

    x_expand_min = x_min_ + np.zeros_like(x_)
    x_expand_max = x_max_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand_min.shape)
    for i in range(n_expand):
        x_expand_min = np.expand_dims(x_expand_min, -1)
        x_expand_max = np.expand_dims(x_expand_max, -1)

    lower_ = np.sum(np.maximum(0, w_l_) * x_expand_min, 1) + np.sum(np.minimum(0, w_l_) * x_expand_max, 1) + b_l_
    upper_ = np.sum(np.maximum(0, w_u_) * x_expand_max, 1) + np.sum(np.minimum(0, w_u_) * x_expand_min, 1) + b_u_

    # import pdb; pdb.set_trace()
    assert_almost_equal(np.clip(lower_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="l_c >y")
    assert_almost_equal(np.clip(y_ - upper_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="u_c <y")


# multi decomposition for convert
def get_standard_values_multid_box_convert(odd=1, dc_decomp=True):

    if dc_decomp:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
            h_0,
            g_0,
        ) = get_standart_values_1d_box(0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
            h_1,
            g_1,
        ) = get_standart_values_1d_box(1, dc_decomp)
    else:
        (
            x_0,
            y_0,
            z_0,
            u_c_0,
            w_u_0,
            b_u_0,
            l_c_0,
            w_l_0,
            b_l_0,
        ) = get_standart_values_1d_box(0, dc_decomp)
        (
            x_1,
            y_1,
            z_1,
            u_c_1,
            w_u_1,
            b_u_1,
            l_c_1,
            w_l_1,
            b_l_1,
        ) = get_standart_values_1d_box(1, dc_decomp)

    if not odd:

        # output (x_0+x_1, x_0+2*x_0) (x_0, x_1)
        x_ = np.concatenate([x_0, x_1], -1)
        z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
        z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
        z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
        y_ = np.concatenate([y_0, y_1], -1)
        b_u_ = np.concatenate([b_u_0, b_u_1], -1)
        u_c_ = np.concatenate([u_c_0, u_c_1], -1)
        b_l_ = np.concatenate([b_l_0, b_l_1], -1)
        l_c_ = np.concatenate([l_c_0, l_c_1], -1)

        if dc_decomp:
            h_ = np.concatenate([h_0, h_1], -1)
            g_ = np.concatenate([g_0, g_1], -1)

        w_u_ = np.zeros((len(x_), 2, 2))
        w_u_[:, 0, 0] = w_u_0[:, 0, 0]
        w_u_[:, 1, 1] = w_u_1[:, 0, 0]

        w_l_ = np.zeros((len(x_), 2, 2))
        w_l_[:, 0, 0] = w_l_0[:, 0, 0]
        w_l_[:, 1, 1] = w_l_1[:, 0, 0]

    else:
        if dc_decomp:
            (
                x_2,
                y_2,
                z_2,
                u_c_2,
                w_u_2,
                b_u_2,
                l_c_2,
                w_l_2,
                b_l_2,
                h_2,
                g_2,
            ) = get_standart_values_1d_box(2, dc_decomp)
        else:
            (x_2, y_2, z_2, u_c_2, w_u_2, b_u_2, l_c_2, w_l_2, b_l_2) = get_standart_values_1d_box(2, dc_decomp)
        x_ = np.concatenate([x_0, x_1, x_2], -1)
        z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0], z_2[:, 0]], -1)
        z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1], z_2[:, 1]], -1)
        z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
        y_ = np.concatenate([y_0, y_1, y_2], -1)
        b_u_ = np.concatenate([b_u_0, b_u_1, b_u_2], -1)
        b_l_ = np.concatenate([b_l_0, b_l_1, b_l_2], -1)
        u_c_ = np.concatenate([u_c_0, u_c_1, u_c_2], -1)
        l_c_ = np.concatenate([l_c_0, l_c_1, l_c_2], -1)

        if dc_decomp:
            h_ = np.concatenate([h_0, h_1, h_2], -1)
            g_ = np.concatenate([g_0, g_1, g_2], -1)

        w_u_ = np.zeros((len(x_), 3, 3))
        w_u_[:, 0, 0] = w_u_0[:, 0, 0]
        w_u_[:, 1, 1] = w_u_1[:, 0, 0]
        w_u_[:, 2, 2] = w_u_2[:, 0, 0]

        w_l_ = np.zeros((len(x_), 3, 3))
        w_l_[:, 0, 0] = w_l_0[:, 0, 0]
        w_l_[:, 1, 1] = w_l_1[:, 0, 0]
        w_l_[:, 2, 2] = w_l_2[:, 0, 0]

    if dc_decomp:
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def assert_output_properties_box_nodc(x_, y_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, name, decimal=5):

    assert np.min(x_min_ <= x_max_), "x_min >x_max for function {}".format(name)

    assert_almost_equal(
        np.clip(x_min_ - x_, 0, np.inf),
        0.0,
        decimal=decimal,
        err_msg="x_min >x_  for function {}".format(name),
    )
    assert_almost_equal(
        np.clip(x_ - x_max_, 0, np.inf),
        0.0,
        decimal=decimal,
        err_msg="x_max < x_  for function {}".format(name),
    )

    x_expand = x_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand.shape)
    for i in range(n_expand):
        x_expand = np.expand_dims(x_expand, -1)

    lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
    upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

    # check that the functions h_ and g_ remains monotonic

    assert_almost_equal(
        np.clip(l_c_ - y_, 0.0, np.inf),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="l_c >y",
    )
    assert_almost_equal(
        np.clip(y_ - u_c_, 0.0, 1e6),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="u_c <y",
    )

    #

    assert_almost_equal(
        np.clip(lower_ - y_, 0.0, np.inf),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="lower_ >y",
    )
    assert_almost_equal(
        np.clip(y_ - upper_, 0.0, 1e6),
        np.zeros_like(y_),
        decimal=decimal,
        err_msg="upper <y",
    )

    # computer lower bounds on the domain

    x_expand_min = x_min_ + np.zeros_like(x_)
    x_expand_max = x_max_ + np.zeros_like(x_)
    n_expand = len(w_u_.shape) - len(x_expand_min.shape)
    for i in range(n_expand):
        x_expand_min = np.expand_dims(x_expand_min, -1)
        x_expand_max = np.expand_dims(x_expand_max, -1)

    lower_ = np.sum(np.maximum(0, w_l_) * x_expand_min, 1) + np.sum(np.minimum(0, w_l_) * x_expand_max, 1) + b_l_
    upper_ = np.sum(np.maximum(0, w_u_) * x_expand_max, 1) + np.sum(np.minimum(0, w_u_) * x_expand_min, 1) + b_u_

    assert_almost_equal(
        np.clip(lower_.min(0) - l_c_.max(0), 0.0, np.inf),
        np.zeros_like(y_.min(0)),
        decimal=decimal,
        err_msg="lower_ >l_c",
    )
    assert_almost_equal(
        np.clip(u_c_.min(0) - upper_.max(0), 0.0, 1e6),
        np.zeros_like(y_.min(0)),
        decimal=decimal,
        err_msg="upper <u_c",
    )
