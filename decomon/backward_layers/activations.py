from __future__ import absolute_import
import warnings
import six
from ..backward_layers.utils import backward_relu_, backward_softplus_
from .utils import V_slope
from tensorflow.keras.layers import Layer
from ..layers import F_FORWARD, F_IBP, F_HYBRID
from ..layers.utils import sigmoid_prime, get_linear_hull_s_shape, softsign_prime, tanh_prime
import tensorflow.keras.backend as K


ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU = "relu_"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"


def backward_relu(
    x,
    dc_decomp=False,
    convex_domain={},
    alpha=0.0,
    max_value=None,
    threshold=0.0,
    slope=V_slope.name,
    mode=F_HYBRID.name,
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

    if dc_decomp:
        raise NotImplementedError()

    if threshold != 0:
        raise NotImplementedError()

    if not (alpha) and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        y = x[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = x[-4:]

        return backward_relu_(y, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, mode=mode, **kwargs)
    raise NotImplementedError()


def backward_sigmoid(inputs, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward  LiRPA of sigmoid
    :param inputs:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    x = inputs[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
    w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
        x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain=convex_domain, mode=mode
    )

    w_u_0 = K.expand_dims(K.expand_dims(w_u_0, 1), -1)
    w_l_0 = K.expand_dims(K.expand_dims(w_l_0, 1), -1)
    b_u_0 = K.expand_dims(K.expand_dims(b_u_0, 1), -1)
    b_l_0 = K.expand_dims(K.expand_dims(b_l_0, 1), -1)
    w_out_u_ = K.maximum(0.0, w_out_u) * w_u_0 + K.minimum(0.0, w_out_u) * w_l_0
    w_out_l_ = K.maximum(0.0, w_out_l) * w_l_0 + K.minimum(0.0, w_out_l) * w_u_0
    b_out_u_ = K.sum(K.maximum(0.0, w_out_u) * b_u_0 + K.minimum(0.0, w_out_u) * b_l_0, 2) + b_out_u
    b_out_l_ = K.sum(K.maximum(0.0, w_out_l) * b_l_0 + K.minimum(0.0, w_out_l) * b_u_0, 2) + b_out_l

    return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]


def backward_tanh(inputs, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward  LiRPA of tanh
    :param inputs:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    x = inputs[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
    w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
        x, func=K.tanh, f_prime=tanh_prime, convex_domain=convex_domain, mode=mode, **kwargs
    )

    # here balance the results for convex linear relaxations
    w_u_ = K.expand_dims(K.expand_dims(w_u_, 1), -1)
    w_l_ = K.expand_dims(K.expand_dims(w_l_, 1), -1)
    b_u_ = K.expand_dims(K.expand_dims(b_u_, 1), -1)
    b_l_ = K.expand_dims(K.expand_dims(b_l_, 1), -1)

    w_u_0 = K.expand_dims(K.expand_dims(w_u_0, 1), -1)
    w_l_0 = K.expand_dims(K.expand_dims(w_l_0, 1), -1)
    b_u_0 = K.expand_dims(K.expand_dims(b_u_0, 1), -1)
    b_l_0 = K.expand_dims(K.expand_dims(b_l_0, 1), -1)
    w_out_u_ = K.maximum(0.0, w_out_u) * w_u_0 + K.minimum(0.0, w_out_u) * w_l_0
    w_out_l_ = K.maximum(0.0, w_out_l) * w_l_0 + K.minimum(0.0, w_out_l) * w_u_0
    b_out_u_ = K.sum(K.maximum(0.0, w_out_u) * b_u_0 + K.minimum(0.0, w_out_u) * b_l_0, 2) + b_out_u
    b_out_l_ = K.sum(K.maximum(0.0, w_out_l) * b_l_0 + K.minimum(0.0, w_out_l) * b_u_0, 2) + b_out_l

    return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]


def bacward_hard_sigmoid(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward  LiRPA of hard sigmoid
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_elu(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward  LiRPA of Exponential Linear Unit
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_selu(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward LiRPA of Scaled Exponential Linear Unit (SELU)
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_linear(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward LiRPA of linear
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    return x


def backward_exponential(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward LiRPAof exponential
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_softplus(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
    """
    Backward LiRPA of softplus
    :param x:
    :param dc_decomp:
    :param convex_domain:
    :param slope:
    :param mode:
    :return:
    """

    if dc_decomp:
        raise NotImplementedError()

    y = x[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = x[-4:]

    return backward_softplus_(y, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain, mode=mode, **kwargs)


def backward_softsign(inputs, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, **kwargs):
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
    if dc_decomp:
        raise NotImplementedError()

    x = inputs[:-4]
    w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]
    w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
        x, func=K.softsign, f_prime=softsign_prime, convex_domain=convex_domain, mode=mode
    )

    w_u_0 = K.expand_dims(K.expand_dims(w_u_0, 1), -1)
    w_l_0 = K.expand_dims(K.expand_dims(w_l_0, 1), -1)
    b_u_0 = K.expand_dims(K.expand_dims(b_u_0, 1), -1)
    b_l_0 = K.expand_dims(K.expand_dims(b_l_0, 1), -1)
    w_out_u_ = K.maximum(0.0, w_out_u) * w_u_0 + K.minimum(0.0, w_out_u) * w_l_0
    w_out_l_ = K.maximum(0.0, w_out_l) * w_l_0 + K.minimum(0.0, w_out_l) * w_u_0
    b_out_u_ = K.sum(K.maximum(0.0, w_out_u) * b_u_0 + K.minimum(0.0, w_out_u) * b_l_0, 2) + b_out_u
    b_out_l_ = K.sum(K.maximum(0.0, w_out_l) * b_l_0 + K.minimum(0.0, w_out_l) * b_u_0, 2) + b_out_l

    return [w_out_u_, b_out_u_, w_out_l_, b_out_l_]


def backward_softmax(x, dc_decomp=False, convex_domain={}, slope=V_slope.name, mode=F_HYBRID.name, axis=-1, **kwargs):
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
    if name == RELU:
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
