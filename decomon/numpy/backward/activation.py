from __future__ import absolute_import

import warnings

import numpy as np
import six
from tensorflow.keras.layers import Layer

from decomon.utils import F_FORWARD, V_slope

from .utils import get_linear_hull_relu, merge_with_previous

ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU = "relu_"
RELU_ = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"


def backward_relu(
    x,
    dc_decomp=False,
    convex_domain=None,
    alpha=0.0,
    max_value=None,
    threshold=0.0,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=False,
    params=None,
    **kwargs,
):
    """
    Backward  LiRPA of relu
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param alpha:
    :param max_value:
    :param threshold:
    :param slope:
    :param mode:
    :return:
    """

    if params is None:
        params = []
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    if threshold != 0:
        raise NotImplementedError()

    if not alpha and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        output = get_linear_hull_relu(x, convex_domain=convex_domain, params=params, **kwargs)

        if previous:
            # TO DO
            return merge_with_previous(output + x[-4:])
        else:
            return output

    raise NotImplementedError()


def backward_sigmoid(
    inputs,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward  LiRPA of sigmoid
    :param inputs:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    raise NotImplementedError()


def backward_tanh(
    inputs,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward  LiRPA of tanh
    :param inputs:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    raise NotImplementedError()


def backward_hard_sigmoid(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward  LiRPA of hard sigmoid
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_elu(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward  LiRPA of Exponential Linear Unit
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_selu(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward LiRPA of Scaled Exponential Linear Unit (SELU)
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_linear(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=False,
    params=None,
    **kwargs,
):
    """
    Backward LiRPA of linear
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if previous:
        z, _, _, _, _, w_b_u, b_b_u, w_b_l, b_b_l = x
        return [z, w_b_u, b_b_u, w_b_l, b_b_l]
    else:
        z, w_f_u, b_f_u = x[:3]
        w_ = np.repeat(np.ones_like(b_f_u[:, None]), z.shape[-1], 1)
        b_ = np.zeros_like(b_f_u)

        return [z, w_, b_, w_, b_]


def backward_exponential(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward LiRPAof exponential
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_softplus(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward LiRPA of softplus
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_softsign(
    inputs,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    params=None,
    **kwargs,
):
    """
    Backward LiRPA of softsign
    :param x:
    :param w_out_u:
    :param b_out_u:
    :param w_out_l:
    :param b_out_l:
    :param convex_domain:
    :param slope: backward slope
    :param mode:
    :return:
    """
    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_softmax(
    x,
    dc_decomp=False,
    convex_domain=None,
    slope=V_slope.name,
    mode=F_FORWARD.name,
    previous=True,
    axis=-1,
    params=None,
    **kwargs,
):
    """
    Backward LiRPA of softmax

    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :param axis:
    :return:
    """

    if convex_domain is None:
        convex_domain = {}
    if params is None:
        params = []
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def deserialize(name):
    """Get the activation from name.

    :param name: name of the method.
    among the implemented Keras activation function.
    :return:

    """
    name = name.lower()

    if name == SOFTMAX:
        return backward_softmax
    if name == ELU:
        return backward_elu
    if name == SELU:
        return backward_selu
    if name == SOFTPLUS:
        return backward_softplus
    if name == SOFTSIGN:
        return backward_softsign
    if name == SIGMOID:
        return backward_sigmoid
    if name == TANH:
        return backward_tanh
    if name in [RELU, RELU_]:
        return backward_relu
    if name == EXPONENTIAL:
        return backward_exponential
    if name == LINEAR:
        return backward_linear
    raise ValueError("Could not interpret " "activation function identifier:", name)


def get(identifier):
    """Get the `identifier` activation function.

    :param identifier: None or str, name of the function.
    :return: The activation function, `linear` if `identifier` is None.
    :raises: ValueError if unknown identifier

    """
    if identifier is None:
        return backward_linear
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, Layer):
            warnings.warn(
                "Do not pass a layer instance (such as {identifier}) as the "
                "activation argument of another layer. Instead, advanced "
                "activation layers should be used just like any other "
                "layer in a model.".format(identifier=identifier.__class__.__name__)
            )
        return identifier
    else:
        raise ValueError("Could not interpret " "activation function identifier:", identifier)
