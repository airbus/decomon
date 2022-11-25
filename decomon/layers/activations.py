from __future__ import absolute_import

import warnings

import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.backend as K

from decomon.layers.core import DecomonLayer

from .core import F_FORWARD, F_HYBRID, F_IBP, StaticVariables
from .utils import (
    exp,
    expand_dims,
    frac_pos,
    get_linear_hull_s_shape,
    get_lower,
    get_upper,
    minus,
    multiply,
    relu_,
    relu_prime,
    sigmoid_prime,
    softplus_,
    softsign_prime,
    sum,
    tanh_prime,
)

ELU = "elu"
SELU = "selu"
SOFTPLUS = "softplus"
SOFTSIGN = "softsign"
SOFTMAX = "softmax"
RELU = "relu"
SIGMOID = "sigmoid"
TANH = "tanh"
EXPONENTIAL = "exponential"
HARD_SIGMOID = "hard_sigmoid"
LINEAR = "linear"
GROUP_SORT_2 = "GroupSort2"


def relu(
    x, dc_decomp=False, convex_domain=None, alpha=0.0, max_value=None, threshold=0.0, mode=F_HYBRID.name, **kwargs
):
    """

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param alpha: see Keras official documentation
    :param max_value: see Keras official documentation
    :param threshold: see Keras official documentation
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    if threshold != 0:
        raise NotImplementedError()

    if not alpha and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        return relu_(x, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, **kwargs)

    raise NotImplementedError()


def linear_hull_s_shape(
    x,
    func=K.sigmoid,
    f_prime=sigmoid_prime,
    dc_decomp=False,
    convex_domain=None,
    mode=F_HYBRID.name,
):
    """
    Computing the linear hull of s-shape functions
    :param x: list of input tensors
    :param func: the function (sigmoid, tanh, softsign...)
    :param f_prime: the derivative of the function (sigmoid_prime...)
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        u_c, l_c = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = func(u_c)
        l_c_ = func(l_c)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        w_u_0, b_u_0, w_l_0, b_l_0 = get_linear_hull_s_shape(
            x, func=func, f_prime=f_prime, convex_domain=convex_domain, mode=mode
        )

        if len(w_u.shape) == len(b_u.shape):
            # it happens with the convert function to spare memory footprint
            n_dim = np.prod(w_u.shape[1:])
            M = np.reshape(
                np.diag([K.cast(1, dtype=K.floatx())] * n_dim), [1, n_dim] + list(w_u.shape[1:])
            )  # usage de numpy pb pour les types
            w_u_ = M * K.concatenate([K.expand_dims(w_u_0, 1)] * n_dim, 1)
            w_l_ = M * K.concatenate([K.expand_dims(w_l_0, 1)] * n_dim, 1)
            b_u_ = b_u_0
            b_l_ = b_l_0

        else:
            w_u_ = K.expand_dims(w_u_0, 1) * w_u  # pour l'instant
            b_u_ = b_u_0 + w_u_0 * b_u
            w_l_ = K.expand_dims(w_l_0, 1) * w_l
            b_l_ = b_l_0 + w_l_0 * b_l

    if mode == F_IBP.name:
        # output = [func(y), x_0, u_c_, l_c_]
        output = [u_c_, l_c_]
    elif mode == F_HYBRID.name:
        # output = [func(y), x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    elif mode == F_FORWARD.name:
        # output = [func(y), x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")

    return output
    # TO DO linear relaxation


def sigmoid(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Sigmoid activation function .
    `1 / (1 + exp(-x))`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    func = K.sigmoid
    f_prime = sigmoid_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def tanh(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Hyperbolic activation function.
    `tanh(x)=2*sigmoid(2*x)+1`

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    func = K.tanh
    f_prime = tanh_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def hard_sigmoid(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Hard sigmoid activation function.
       Faster to compute than sigmoid activation.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def elu(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Exponential linear unit.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def selu(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def linear(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA foe Linear (i.e. identity) activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """
    if convex_domain is None:
        convex_domain = {}
    return x


def exponential(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Exponential activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """

    if convex_domain is None:
        convex_domain = {}
    return exp(x, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, **kwargs)


def softplus(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Softplus activation function `log(exp(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    return softplus_(x, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def softsign(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """LiRPA for Softsign activation function `x / (abs(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """

    if convex_domain is None:
        convex_domain = {}
    func = K.softsign
    f_prime = softsign_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def softmax(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, axis=-1, clip=True, **kwargs):
    """LiRPA for Softmax activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: type of convex input domain (None or dict)
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: see Keras official documentation
    :return: the updated list of tensors

    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    x_ = minus(x, mode=mode)
    x_0 = exponential(x_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    x_sum = sum(x_0, axis=axis, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    x_frac = expand_dims(
        frac_pos(x_sum, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode), mode=mode, axis=axis
    )

    x_final = multiply(x_0, x_frac, mode=mode, convex_domain=convex_domain)

    if mode == F_IBP.name:
        u_c, l_c = x_final
        if clip:
            return [K.minimum(u_c, 1.0), K.maximum(l_c, 0.0)]
        else:
            return x_final
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x_final
        if clip:
            u_c = K.minimum(u_c, 1.0)
            l_c = K.maximum(l_c, 0.0)
        return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]

    return x_final


def group_sort_2(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, data_format="channels_last", **kwargs):

    if convex_domain is None:
        convex_domain = {}
    raise NotImplementedError()


def deserialize(name):
    """Get the activation from name.

    :param name: name of the method.
    among the implemented Keras activation function.
    :return: the activation function

    """
    name = name.lower()

    if name == SOFTMAX:
        return softmax
    elif name == ELU:
        return elu
    elif name == SELU:
        return selu
    elif name == SOFTPLUS:
        return softplus
    elif name == SOFTSIGN:
        return softsign
    elif name == SIGMOID:
        return sigmoid
    elif name == TANH:
        return tanh
    elif name == RELU:
        return relu_
    elif name == EXPONENTIAL:
        return exponential
    elif name == LINEAR:
        return linear
    elif name == GROUP_SORT_2:
        return group_sort_2
    else:
        raise ValueError(f"Could not interpret activation function identifier: {name}")


def get(identifier):
    """Get the `identifier` activation function.

    :param identifier: None or str, name of the function.
    :return: The activation function, `linear` if `identifier` is None.
    :raises: ValueError if unknown identifier

    """
    if identifier is None:
        return linear
    elif isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    elif callable(identifier):
        if isinstance(identifier, DecomonLayer):
            warnings.warn(
                "Do not pass a layer instance (such as {identifier}) as the "
                "activation argument of another layer. Instead, advanced "
                "activation layers should be used just like any other "
                "layer in a model.".format(identifier=identifier.__class__.__name__)
            )
            return identifier
        import pdb

        pdb.set_trace()
    else:
        raise ValueError("Could not interpret " "activation function identifier:", identifier)
