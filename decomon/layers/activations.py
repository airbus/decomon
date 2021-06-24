from __future__ import absolute_import
from .utils import relu_, get_upper, get_lower, get_linear_hull_s_shape
import warnings
import six
from decomon.layers.core import DecomonLayer
from .core import F_HYBRID, F_FORWARD, F_IBP
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


def relu(
    x,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    alpha=0.0,
    max_value=None,
    threshold=0.0,
    mode=F_HYBRID.name,
    fast=True,
):
    """Rectified Linear Unit.
    With default values, it returns element-wise `max(x, 0)`.
    Otherwise, it follows:
    `f(x) = max_value` for `x >= max_value`,
    `f(x) = x` for `threshold <= x < max_value`,
    `f(x) = alpha * (x - threshold)` otherwise.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :param alpha:
    :param max_value:
    :param threshold:
    :return: the updated list of tensors

    """
    if threshold != 0:
        raise NotImplementedError()

    if not (alpha) and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        return relu_(x, dc_decomp=dc_decomp, grad_bounds=grad_bounds, convex_domain=convex_domain, mode=mode, fast=fast)

    raise NotImplementedError()


# Lagrangian Decomposition for Neural Network Verification
def sigmoid_prime(x):
    s_x = K.sigmoid(x)
    return s_x * (1 - s_x)


def tanh_prime(x):
    s_x = K.tanh(x)
    return 1 - K.pow(s_x, 2)


def softsign_prime(x):
    return 1.0 / K.pow(1.0 + K.abs(x), 2)


"""
def linear_hull_upper(lower, upper, func=K.sigmoid, f_prime=sigmoid_prime):
    s_u_prime = f_prime(upper)
    s_u = func(upper)
    s_l = func(lower)
    if s_u_prime >= (s_u - s_l) / (upper - lower):
        w_u = (s_u - s_l) / (upper - lower)
        b_u = -w_u * lower + s_l
    else:
        if lower >= 0:
            w_u = 0.
            b_u = s_u
        else:
            t_ = np.linspace(0., 1., 100) * upper
            s_p_t_ = f_prime(t_)
            score = s_p_t_ - (func(t_) - s_l) / (t_ - lower)

            index_0 = np.argmin(np.abs(score))
            if score[index_0] <= 0:
                index_0 -= 1
            s_t = func(t_[index_0])
            t = t_[index_0]

            w_u = (s_t - s_l) / (t - lower)
            b_u = -w_u * lower + s_l

            V_0 = (w_u * upper + b_u - s_t) * (upper - t) / 2.
            # V_1 = (s_t-s_l)*(-t-lower)/2.
            V_2 = (s_t - s_l) * (t - lower) / 2.
            print(V_2, V_0)
            if V_2 <= V_0:
                w_u = 0.
                b_u = s_u

    return [w_u, b_u]
"""


def get_t_upper(u_c_flat, l_c_flat, s_l, func=K.sigmoid, f_prime=sigmoid_prime):
    """
    linear interpolation between lower and upper bounds on the sigmoid to have a symbolic approximation of the best
    coefficient for the affine upper bound
    :param u_c:
    :param l_c:
    :param s_u:
    :param s_l:
    :return:
    """
    # step1: find t
    t_ = np.linspace(0, 1.0, 100)[None, None, :] * u_c_flat

    s_p_t_ = f_prime(t_)
    s_t_ = func(t_)

    score = s_p_t_ - (s_t_ - K.expand_dims(s_l, -1)) / (t_ - l_c_flat)
    titi = K.clip(1.0 / score, -1e6, 1e6)
    index = K.cast(K.cast(K.equal(titi, K.expand_dims(K.min(titi, -1), -1)), "int32"), "float32")
    score_ = (s_t_ - K.expand_dims(s_l, -1)) / (t_ - l_c_flat)

    w_u_1 = K.sum(score_ * index, -1)  # attention !!!!

    s_t_final = K.sum(s_t_ * index, -1)
    t_final = K.sum(t_ * index, -1)

    V_0 = (w_u_1 * (u_c_flat[:, :, 0] - l_c_flat[:, :, 0]) + s_l - s_t_final) * (u_c_flat[:, :, 0] - t_final)
    V_2 = (s_t_final - s_l) * (t_final - l_c_flat[:, :, 0])

    return [w_u_1, V_0, V_2]


def get_t_lower(u_c_flat, l_c_flat, s_u, func=K.sigmoid, f_prime=sigmoid_prime):
    """
    linear interpolation between lower and upper bounds on the sigmoid to have a symbolic approximation of the best
    coefficient for the affine lower bound
    :param u_c:
    :param l_c:
    :param s_u:
    :param s_l:
    :return:
    """
    # step1: find t
    t_ = np.linspace(0, 1.0, 100)[None, None, :] * l_c_flat

    s_p_t_ = f_prime(t_)
    s_t_ = func(t_)

    score = s_p_t_ - (K.expand_dims(s_u, -1) - s_t_) / (u_c_flat - t_)
    titi = K.clip(-1.0 / score, -1e6, 1e6)
    index = K.cast(K.cast(K.equal(titi, K.expand_dims(K.min(titi, -1), -1)), "int32"), "float32")

    score_ = (K.expand_dims(s_u, -1) - s_t_) / (u_c_flat - t_)

    w_l_1 = K.sum(score_ * index, -1)  # attention !!!!

    s_t_final = K.sum(s_t_ * index, -1)
    t_final = K.sum(t_ * index, -1)

    V_0 = (w_l_1 * (u_c_flat[:, :, 0] - l_c_flat[:, :, 0]) + s_t_final - s_u) * (t_final - l_c_flat[:, :, 0])
    V_2 = (s_u - s_t_final) * (u_c_flat[:, :, 0] - t_final)

    return [w_l_1, V_0, V_2]


def linear_hull_s_shape(
    x,
    func=K.sigmoid,
    f_prime=sigmoid_prime,
    dc_decomp=False,
    grad_bounds=False,
    convex_domain={},
    mode=F_HYBRID.name,
    fast=True,
):

    if dc_decomp or grad_bounds:
        raise NotImplementedError()
    if mode == F_IBP.name:
        y, x_0, u_c, l_c = x[:4]
    if mode == F_HYBRID.name:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:8]
    if mode == F_FORWARD.name:
        y, x_0, w_u, b_u, w_l, b_l = x[:6]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = func(u_c)
        l_c_ = func(l_c)

        output = [y, x_0, u_c_, l_c_]

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        # upper bound
        # derivative
        s_u_prime = f_prime(u_c)
        s_l_prime = f_prime(l_c)
        s_u = func(u_c)
        s_l = func(l_c)

        """
        shape = u_c.shape[1:]
        u_c_flat = K.reshape(u_c, (-1, np.prod(shape), 1))
        l_c_flat = K.reshape(l_c, (-1, np.prod(shape), 1))

        # get upper bound

        w_u_0 = K.reshape(K.switch(K.equal(u_c, l_c), s_u - s_l, (s_u - s_l) / (u_c - l_c)), [-1] + list(u_c.shape[1:]))

        w_u_1, V_0_u, V_2_u = get_t_upper(u_c_flat, l_c_flat, s_l, func=func, f_prime=f_prime)

        # V_0 = (w_u_1*(u_c_flat[:,:,0] - l_c_flat[:,:, 0]) + s_l - s_t_final)*(u_c_flat[:,:,0]-t_final)
        # V_2 = (s_t_final - s_l)*(t_final - l_c_flat[:,:,0])

        w_u_1_ = K.reshape(K.switch(K.greater(V_0_u, V_2_u), w_u_1, 0 * w_u_1), [-1] + list(u_c.shape[1:]))

        w_u_1 = K.switch(K.greater(l_c, -K.epsilon()), 0 * w_u_0, w_u_1_)
        w_u_1_ = K.reshape(w_u_1, [-1] + list(u_c.shape[1:]))

        threshold_u = s_u_prime - (s_u - s_l) / (K.maximum(K.epsilon(), u_c - l_c))
        w_u_a_ = K.reshape(K.switch(K.greater(threshold_u, 0.0), w_u_0, w_u_1_), [-1] + list(w_u_0.shape[1:]))
        # if collision to a single point
        w_u_a_ = K.switch(K.equal(u_c, l_c), 0 * w_u_0, w_u_a_)
        b_u_a_ = K.switch(
            K.greater(w_u_a_, 0.0),
            -K.expand_dims(K.sum(K.reshape(w_u_a_, [-1, np.prod(shape), 1]) * l_c_flat, (-1, -2)), 1) + s_l,
            s_u,
        )
        """
        w_u_a_ = 0 * s_u
        b_u_a_ = s_u
        w_u_ = K.expand_dims(w_u_a_, 1) * w_u
        b_u_ = w_u_a_ * b_u + b_u_a_

        # get lower bound
        """
        w_l_0 = w_u_0
        w_l_1, V_0_l, V_2_l = get_t_lower(u_c_flat, l_c_flat, s_u, func=func, f_prime=f_prime)
        w_l_1_ = K.reshape(K.switch(K.greater(V_0_l, V_2_l), w_l_1, 0 * w_l_1), [-1] + list(l_c.shape[1:]))
        w_l_1 = K.switch(K.greater(u_c, K.epsilon()), w_l_1, 0 * w_l_0)
        # w_l_1_ = K.reshape(K.switch(K.greater(V_0_l, V_2_l), w_l_1, 0 * w_l_1), [-1] + list(l_c.shape[1:]))
        w_l_1_ = K.reshape(w_l_1, [-1] + list(l_c.shape[1:]))

        threshold_l = s_l_prime - (s_u - s_l) / (K.maximum(K.epsilon(), u_c - l_c))
        w_l_a_ = K.reshape(K.switch(K.greater(threshold_l, 0.0), w_l_0, w_l_1_), [-1] + list(w_l_0.shape[1:]))
        w_l_a_ = K.switch(K.equal(u_c, l_c), 0 * w_l_0, w_l_a_)
        b_l_a_ = K.switch(
            K.greater(w_l_a_, 0.0),
            -K.expand_dims(K.sum(K.reshape(w_l_a_, [-1, np.prod(shape), 1]) * u_c_flat, (-1, -2)), 1) + s_u,
            s_l,
        )
        """
        w_l_a_ = 0 * s_l
        b_l_a_ = s_l
        w_l_ = K.expand_dims(w_l_a_, 1) * w_l
        b_l_ = w_l_a_ * b_l + b_l_a_

    if mode == F_IBP.name:
        output = [func(y), x_0, u_c_, l_c_]

    if mode == F_HYBRID.name:
        output = [func(y), x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    if mode == F_FORWARD.name:
        output = [func(y), x_0, w_u_, b_u_, w_l_, b_l_]

    return output
    # TO DO linear relaxation
    raise NotImplementedError()


def sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Sigmoid activation function .

    `1 / (1 + exp(-x))`.
    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    func = K.sigmoid
    f_prime = sigmoid_prime
    return linear_hull_s_shape(
        x, func, f_prime, dc_decomp=dc_decomp, grad_bounds=grad_bounds, convex_domain={}, mode=mode, fast=fast
    )


def tanh(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Hyperbolic activation function.

    `tanh(x)=2*sigmoid(2*x)+1`

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    func = K.tanh
    f_prime = tanh_prime
    return linear_hull_s_shape(
        x, func, f_prime, dc_decomp=dc_decomp, grad_bounds=grad_bounds, convex_domain={}, mode=mode, fast=fast
    )


def hard_sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Hard sigmoid activation function.
       Faster to compute than sigmoid activation.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def elu(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Exponential linear unit.

    Fast and Accurate Deep Network Learning
    by Exponential  Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
    Relu(x) - Relu(-alpha*[exp(x)-1])

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates whether
    we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def selu(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
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
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def linear(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Linear (i.e. identity) activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    return x


def exponential(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Exponential activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def softplus(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Softplus activation function `log(exp(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
     whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def softsign(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True):
    """Softsign activation function `x / (abs(x) + 1)`.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """

    func = K.softsign
    f_prime = softsign_prime
    return linear_hull_s_shape(
        x, func, f_prime, dc_decomp=dc_decomp, grad_bounds=grad_bounds, convex_domain={}, mode=mode, fast=fast
    )


def softmax(x, dc_decomp=False, grad_bounds=False, convex_domain={}, mode=F_HYBRID.name, fast=True, axis=-1):
    """Softmax activation function.

    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param axis: integer or -1 that indicates
    on which axis we operate the softmax
    :param convex_domain: the type of convex domain (see ???)
    :return: the updated list of tensors

    """
    if dc_decomp or grad_bounds:
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
    else:
        raise ValueError("Could not interpret " "activation function identifier:", identifier)
