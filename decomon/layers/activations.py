from __future__ import absolute_import
from .utils import (
    relu_,
    softplus_,
    get_upper,
    get_lower,
    get_linear_hull_s_shape,
    sigmoid_prime,
    softsign_prime,
    tanh_prime,
    relu_prime,
    minus,
    frac_pos,
    sum,
    multiply,
    expand_dims
)
import warnings
import six
from decomon.layers.core import DecomonLayer
from .core import F_HYBRID, F_FORWARD, F_IBP, StaticVariables
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


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
GROUP_SORT_2 = 'GroupSort2'

def relu(x, dc_decomp=False, convex_domain={}, alpha=0.0, max_value=None, threshold=0.0, mode=F_HYBRID.name, **kwargs):
    """Rectified Linear Unit.
    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param alpha: see Keras official documentation
    :param max_value: see Keras official documentation
    :param threshold: see Keras official documentation
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """
    if threshold != 0:
        raise NotImplementedError()

    if not (alpha) and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        return relu_(x, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, **kwargs)

    raise NotImplementedError()


def linear_hull_s_shape(
    x,
    func=K.sigmoid,
    f_prime=sigmoid_prime,
    dc_decomp=False,
    convex_domain={},
    mode=F_HYBRID.name,
):
    """
    Computing the linear hull of s-shape functions
    :param x: list of input tensors
    :param func: the function (sigmoid, tanh, softsign...)
    :param f_prime: the derivative of the function (sigmoid_prime...)
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :return: the updated list of tensors
    """

    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        # y, x_0, u_c, l_c = x[:4]
        u_c, l_c = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    if mode == F_HYBRID.name:
        # y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
    if mode == F_FORWARD.name:
        # y, x_0, w_u, b_u, w_l, b_l = x[:6]
        x_0, w_u, b_u, w_l, b_l = x[: StaticVariables(dc_decomp=dc_decomp, mode=mode).nb_tensors]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = func(u_c)
        l_c_ = func(l_c)

        # output = [y, x_0, u_c_, l_c_]

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

    if mode == F_HYBRID.name:
        # output = [func(y), x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    if mode == F_FORWARD.name:
        # output = [func(y), x_0, w_u_, b_u_, w_l_, b_l_]
        output = [x_0, w_u_, b_u_, w_l_, b_l_]

    return output
    # TO DO linear relaxation


def sigmoid(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Sigmoid activation function .
    `1 / (1 + exp(-x))`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors
    """

    func = K.sigmoid
    f_prime = sigmoid_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def tanh(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Hyperbolic activation function.
    `tanh(x)=2*sigmoid(2*x)+1`

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors
    """

    func = K.tanh
    f_prime = tanh_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def hard_sigmoid(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Hard sigmoid activation function.
       Faster to compute than sigmoid activation.
    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """

    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def elu(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Exponential linear unit.

    Fast and Accurate Deep Network Learning
    by Exponential  Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    Relu(x) - Relu(-alpha*[exp(x)-1])

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors


    """
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def selu(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Scaled Exponential Linear Unit (SELU).

    SELU is equal to: `scale * elu(x, alpha)`, where alpha and scale
    are predefined constants. The values of `alpha` and `scale` are
    chosen so that the mean and variance of the inputs are preserved
    between two consecutive layers as long as the weights are initialized
    correctly (see `lecun_normal` initialization) and the number of inputs
    is "large enough" (see references for more information).

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """
    if dc_decomp:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def linear(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Linear (i.e. identity) activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """
    return x


def exponential(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Exponential activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """
    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        return [K.exp(e) for e in x]
    if mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c =  get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

    u_c_ = K.exp(u_c)
    l_c_ = K.exp(l_c)

    y = (u_c+l_c)/2. # do finetuneting
    slope = K.exp(y)

    w_u_0 = (u_c_ - l_c_)/K.maximum(u_c-l_c, K.epsilon())
    b_u_0 = l_c_ - w_u_0*l_c

    w_l_0 = K.exp(y)
    b_l_0 = w_l_0*(1-y)

    w_u_ = w_u_0[:,None]*w_u
    b_u_ = b_u_0*b_u + b_u_0
    w_l_ = w_l_0[:, None] * w_l
    b_l_ = b_l_0 * b_l + b_l_0

    if mode == F_FORWARD.name:
        return [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def softplus(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Softplus activation function `log(exp(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """
    if dc_decomp:
        raise NotImplementedError()

    return softplus_(x, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    # TO DO linear relaxation
    raise NotImplementedError()


def softsign(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, **kwargs):
    """Softsign activation function `x / (abs(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """

    func = K.softsign
    f_prime = softsign_prime
    return linear_hull_s_shape(x, func, f_prime, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def softmax(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, axis=-1, clip=True, **kwargs):
    """Softmax activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex input domain
    :param mode: type of Forward propagation (IBP, Forward or Hybrid)
    :param kwargs: extra parameters
    :return: the updated list of tensors

    """
    if dc_decomp:
        raise NotImplementedError()

    x_ = minus(x, mode=mode)
    x_0 = exponential(x_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    x_sum = sum(x_0, axis=axis, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    x_frac = expand_dims(frac_pos(x_sum, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode), mode=mode, axis=axis)

    x_final = multiply(x_0, x_frac, mode=mode, convex_domain=convex_domain)

    if mode==F_IBP.name:
        u_c, l_c = x_final
        if clip:
            return [K.minimum(u_c, 1.), K.maximum(l_c, 0.)]
        else:
            return x_final
    if mode==F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x_final
        if clip:
            u_c = K.minimum(u_c, 1.)
            l_c = K.maximum(l_c, 0.)
        return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]

    return x_final


    # TO DO linear relaxation
    raise NotImplementedError()

def group_sort_2(x, dc_decomp=False, convex_domain={}, mode=F_HYBRID.name, data_format="channels_last", **kwargs):

    input_shape = x[-1].shape
    if data_format=="channels_last":
        axis=-1
        # check dimension
        if input_shape[-1]%2!=0:
            raise ValueError()
    else:
        axis=1
        if input_shape[1]%2!=0:
            raise ValueError()

    import pdb; pdb.set_trace()






def deserialize(name):
    """Get the activation from name.

    :param name: name of the method.
    among the implemented Keras activation function.
    :return: the activation function

    """
    name = name.lower()

    if name == SOFTMAX:
        return softmax
    if name == ELU:
        return elu
    if name == SELU:
        return selu
    if name == SOFTPLUS:
        return softplus
    if name == SOFTSIGN:
        return softsign
    if name == SIGMOID:
        return sigmoid
    if name == TANH:
        return tanh
    if name == RELU:
        return relu_
    if name == EXPONENTIAL:
        return exponential
    if name == LINEAR:
        return linear
    if name ==GROUP_SORT_2:
        return group_sort_2
    raise ValueError("Could not interpret " "activation function identifier:", name)


def get(identifier):
    """Get the `identifier` activation function.

    :param identifier: None or str, name of the function.
    :return: The activation function, `linear` if `identifier` is None.
    :raises: ValueError if unknown identifier

    """
    if identifier is None:
        return linear
    if isinstance(identifier, six.string_types):
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
