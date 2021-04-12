from __future__ import absolute_import
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
import tensorflow.keras as keras
import numpy as np
from .core import Ball, Box, Vertex


def _get_shape_tuple(init_tuple, tensor, start_idx, int_shape=None):
    """Finds non-specific dimensions in the static shapes.
    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.
    Arguments:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape
    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.

    sources: official Keras directory
    """
    # replace all None in int_shape by K.shape
    if int_shape is None:
        int_shape = K.int_shape(tensor)[start_idx:]
    if not any(not s for s in int_shape):
        return init_tuple + tuple(int_shape)
    shape = K.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
        if not s:
            int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)


##############
# SYMBOLIC UPPER/ LOWER BOUNDS
# compute symbolically constant upper and lower
# with the current knowledge of the convex domain considered
##############


# case 1: a box
def get_upper_box(x_min, x_max, w, b):
    """
    #compute the max of an affine function
    within a box (hypercube) defined by its extremal corners
    :param x_min: lower bound of the box domain
    :param x_max: upper bound of the box domain
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: max_(x >= x_min, x<=x_max) w*x + b
    """

    # split into positive and negative components
    w_pos = K.maximum(w, 0.0)
    w_neg = K.minimum(w, 0.0)

    x_min_ = x_min + 0 * x_min
    x_max_ = x_max + 0 * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_max_ + w_neg * x_min_, 1) + b


def get_lower_box(x_min, x_max, w, b):
    """

    :param x_min: lower bound of the box domain
    :param x_max: upper bound of the box domain
    :param w_l: weights of the affine lower bound
    :param b_l: bias of the affine lower bound
    :return: the lowest value output by the affine function on the domain
    """

    w_pos = K.maximum(w, 0.0)
    w_neg = K.minimum(w, 0.0)

    x_min_ = x_min + 0 * x_min
    x_max_ = x_max + 0 * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_min_ + w_neg * x_max_, 1) + b


# case 2 : a ball
def get_lq_norm(x, p, axis=-1):
    """
    compute Lp norm (p=1 or 2)
    :param x: tensor
    :param p: the power must be an integer in (1, 2)
    :param axis: the axis on which we compute the norm
    :return: ||w||^p
    """

    if p not in [1, 2]:
        raise NotImplementedError()

    if p == 1:
        # x_p = K.sum(K.abs(x), axis)
        x_q = K.max(K.abs(x), axis)
    if p == 2:
        x_q = K.sqrt(K.sum(K.pow(x, p), axis))

    return x_q


def get_upper_ball(x_0, eps, p, w, b):
    """
    max of an affine function over an Lp ball
    :param x_0: the center of the ball
    :param eps: the radius
    :param p: the type of Lp norm considered
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: max_(|x - x_0|_p<= eps) w*x + b
    """

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_upper_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        upper = eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + upper


def get_lower_ball(x_0, eps, p, w, b):
    """
    min of an affine fucntion over an Lp ball
    :param x_0: the center of the ball
    :param eps: the radius
    :param p: the type of Lp norm considered
    :param w: weights of the affine function
    :param b: bias of the affine function
    :return: min_(|x - x_0|_p<= eps) w*x + b
    """
    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_lower_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        lower = -eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + lower


# other possibilities: use gradient information
# to refine the upper and lower bounds
# (either with Lipschitz or monotony)
def get_upper(x, w, b, convex_domain={}):
    """
    Meta function that aggregates all the way
    to compute a constant upper bounds depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """

    if convex_domain is None or len(convex_domain) == 0:
        # box
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b)

    if convex_domain["name"] == Ball.name:

        eps = convex_domain["eps"]
        p = convex_domain["p"]
        return get_upper_ball(x, eps, p, w, b)

    if convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    raise NotImplementedError()


def get_lower(x, w, b, convex_domain={}):
    """
     Meta function that aggregates all the way
     to compute a constant lower bound depending on the convex domain
    :param x: the tensors that represent the domain
    :param w: the weights of the affine function
    :param b: the bias
    :param convex_domain: the type of convex domain (see ???)
    :return: a constant upper bound of the affine function
    """
    if len(convex_domain) == 0:
        # box
        x_min = x[:, 0]
        x_max = x[:, 1]

        return get_lower_box(x_min, x_max, w, b)

    if convex_domain["name"] == Box.name:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min, x_max, w, b)

    if convex_domain["name"] == Ball.name:

        eps = convex_domain["eps"]
        p = convex_domain["p"]

        return get_lower_ball(x, eps, p, w, b)

    if convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    raise NotImplementedError()


#####
# USE SYMBOLIC GRADIENT DESCENT WITH OVERESTIMATION GUARANTEES
#####
# first compute gradient of the function
def get_grad(x, constant, W, b):
    """
    We compute the gradient of the function f at sample x

    f = sum_{i<= n_linear} max(constant_i, W_i*x + b_i)
    it is quite easy to compute the gradient symbolically, without using the gradient operator
    it is either 0 if f(x) = constant or W else
    this trick allows to be backpropagation compatible

    :param x: Keras Tensor (None, n_dim, ...)
    :param constant: Keras Tensor (None, n_linear, ...)
    :param W: Keras Tensor (None, n_dim, n_linear, ...)
    :param b: Keras Tensor (None, n_linear, ...)
    :return: Keras Tensor the gradient of the function (None, n_dim, ...)
    """
    # product W*x + b
    # import pdb; pdb.set_trace()
    # x_ = K.expand_dims(x, 2)  # (None, n_dim, 1, 1)
    # z = K.sum(W * x_, 1) + b

    x_ = K.expand_dims(x, -2)
    # y_ = K.expand_dims(x, 2)
    # import pdb; pdb.set_trace()
    z = K.sum(W * x_, 1) + b

    grad_ = K.sum(K.expand_dims(-K.sign(constant - K.maximum(constant, z)), 1) * W, 2)  # (None, n_dim, ...)

    return grad_


def compute_L(W):
    """
    We compute the largest possible norm of the gradient
    :param W: Keras Tensor (None, n_dim, n_linear, ...)
    :return: Keras Tensor with an upper bound on the largest magnitude of the gradient
    """
    # do L2 norm
    return K.sum(K.sqrt(K.sum(K.pow(W, 2), 1)), 1)


def compute_R(z, convex_domain):
    """
    We compute the largest L2 distance of the starting point with the global optimum
    :param z: Keras Tensor
    :param convex_domain: Dictionnary to complement z on the convex domain
    :return: Keras Tensor an upper bound on the distance
    """

    if len(convex_domain) == 0:
        # compute the L2 distance z[:, 0], z[:, 1]
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Box.name:  # to improve
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        dist_ = K.sqrt(K.sum(K.pow(z - z + convex_domain["eps"], 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        dist_ = convex_domain["eps"] * K.ones_like(z)
    else:
        raise NotImplementedError()

    return dist_


def get_start_point(z, convex_domain):
    """
    Create a warm start for the optimization (the mid point to minimize the largest distance between the
    warm start and the global optimum
    :param z: Keras Tensor
    :param convex_domain: Dictionnary to complement z on the convex domain
    :return: Keras Tensor
    """

    if len(convex_domain) == 0:
        # compute the L2 distance z[:, 0], z[:, 1]
        return (z[:, 0] + z[:, 1]) / 2.0
    elif convex_domain["name"] == Box.name:  # to improve
        return (z[:, 0] + z[:, 1]) / 2.0
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        return z
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        return z
    else:
        raise NotImplementedError()


def get_coeff_grad(R, k, g):
    """

    :param R: Keras Tensor that reprends the largest distance to the gloabl optimum
    :param k: the number of iteration done so far
    :param g: the gradient
    :return: the adaptative step size for the gradient
    """

    denum = np.sqrt(k) * K.sqrt(K.sum(K.pow(g, 2), 1))
    return R[:, None] / K.maximum(K.epsilon(), denum)


def grad_descent_conv(z, concave_upper, convex_lower, op_pos, ops_neg, n_iter):
    """

    :param z:
    :param concave_upper:
    :param convex_lower:
    :param op_pos:
    :param ops_neg:
    :param n_iter:
    :return:
    """

    raise NotImplementedError()


def grad_descent(z, convex_0, convex_1, convex_domain, n_iter=5):
    """

    :param z: Keras Tensor
    :param constant: Keras Tensor, the constant of each component
    :param W: Keras Tensor, the affine of each component
    :param b: Keras Tensor, the bias of each component
    :param convex_domain: Dictionnary to complement z on the convex domain
    :param n_iter: the number of total iteration
    :return:
    """

    constant_0, W_0, b_0 = convex_0
    constant_1, W_1, b_1 = convex_1
    # x_k = K.expand_dims(z[:, 0], -1)

    # init
    # x_k = get_start_point(z, convex_domain)
    x_k = K.expand_dims(get_start_point(z, convex_domain), -1)
    R = compute_R(z, convex_domain)

    # g_k = get_grad(x_k, constant, W, b)
    # alpha_k = get_coeff_grad(R, 1, g_k)
    # x_k = x_k - K.expand_dims(alpha_k, 1) * g_k  # (None, 784, 1)
    # k = 1
    for k in range(1, n_iter + 1):
        g_k_0 = get_grad(x_k, constant_0, W_0, b_0)
        g_k_1 = get_grad(x_k, constant_1, W_1, b_1)
        g_k = g_k_0 + g_k_1
        alpha_k = get_coeff_grad(R, k + 1, g_k)
        x_k = x_k - K.expand_dims(alpha_k, 1) * g_k
        # project on the domain

    # evaluate f(x_k)
    # check convergence
    mask_grad = K.sign(K.sum(K.pow(get_grad(x_k, constant_0, W_0, b_0) + get_grad(x_k, constant_1, W_1, b_1), 2), 1))
    # 1 -> we have not yet converged
    # 0 -> we have converged
    # import pdb; pdb.set_trace()
    x_k = K.expand_dims(x_k, -2)
    z_0 = K.sum(W_0 * x_k, 1) + b_0  # (None, 10, 1)
    z_1 = K.sum(W_1 * x_k, 1) + b_1
    # f_x_k = K.sum(K.maximum(constant_0, z_0), 1) + K.sum(K.maximum(constant_1, z_1), 1)
    f_x_k = K.sum(K.maximum(constant_0, z_0) + K.maximum(constant_1, z_1), 1)
    L = compute_L(K.concatenate([W_0, W_1], 1))
    worst_case = (L * R[:, None]) / np.sqrt(max(1, n_iter))

    return f_x_k - mask_grad * worst_case


class NonPos(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w):
        return w * K.cast(K.less_equal(w, 0.0), K.floatx())


class MultipleConstraint(Constraint):
    """"""

    def __init__(self, constraint_0, constraint_1, **kwargs):
        super(MultipleConstraint, self).__init__(**kwargs)
        if constraint_0:
            self.constraints = [constraint_0, constraint_1]
        else:
            self.constraints = [constraint_1]

    def __call__(self, w):
        w_ = w
        for c in self.constraints:
            w_ = c.__call__(w_)

        return w_


class Project_initializer_pos(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer, **kwargs):
        super(Project_initializer_pos, **kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None):
        w_ = self.initializer.__call__(shape, dtype)
        return K.maximum(0.0, w_)


class Project_initializer_neg(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer, **kwargs):
        super(Project_initializer_neg, **kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None):
        w_ = self.initializer.__call__(shape, dtype)
        return K.minimum(0.0, w_)


def relu_(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
    """
    :param x: list of input tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :return: the updated list of tensors
    """

    if grad_bounds:
        raise NotImplementedError()

    if dc_decomp:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = x
        h_ = K.maximum(h, -g)
        g_ = g
    else:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

    # compute upper and lower bounds
    # and keep the minimum between the constant and the computed upper bounds
    # and keep the maximum between the constant and the computer lower bounds
    upper = K.minimum(u_c, get_upper(x_0, w_u, b_u, convex_domain))
    lower = K.maximum(l_c, get_lower(x_0, w_l, b_l, convex_domain))

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper) - 1, -1, 0)
    index_linear = K.clip(K.sign(lower) + 1, 0, 1)

    # update the new upper and lower constant bounds after relu
    u_c_ = upper
    l_c_ = K.maximum(0.0, lower)

    if dc_decomp:
        h_ = (1 - index_dead) * h_
        g_ = (1 - index_dead) * g_

    # 1 if upper<=-lower else 0
    index_a = -K.clip(K.sign(upper + lower) - 1, -1.0, 0.0)

    # 1 if upper>-lower else 0
    index_b = K.ones_like(index_a) - index_a

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    alpha = upper / K.maximum(K.epsilon(), upper - lower)
    # scaling factor for the upper bound on the relu
    # see README

    w_u_ = K.expand_dims(alpha, 1) * w_u
    b_u_ = alpha * (b_u - lower)

    w_l_b = w_l
    b_l_b = b_l

    w_l_a = 0 * w_l
    b_l_a = 0 * b_l

    w_l_ = K.expand_dims(index_a, 1) * w_l_a + K.expand_dims(index_b, 1) * w_l_b
    b_l_ = index_a * b_l_a + index_b * b_l_b

    # set everything to zero if the relu is inactive
    w_u_ = K.expand_dims(1 - index_dead, 1) * w_u_
    b_u_ = (1 - index_dead) * b_u_

    w_l_ = K.expand_dims(1 - index_dead, 1) * w_l_
    b_l_ = (1 - index_dead) * b_l_

    u_c_ = (1 - index_dead) * u_c_
    l_c_ = (1 - index_dead) * l_c_

    # set everything to the initial state if relu_ is linear
    w_u_ = K.expand_dims(1 - index_linear, 1) * w_u_ + K.expand_dims(index_linear, 1) * w_u
    b_u_ = (1 - index_linear) * b_u_ + index_linear * b_u
    u_c_ = (1 - index_linear) * u_c_ + index_linear * u_c

    w_l_ = K.expand_dims(1 - index_linear, 1) * w_l_ + K.expand_dims(index_linear, 1) * w_l
    b_l_ = (1 - index_linear) * b_l_ + index_linear * b_l
    l_c_ = (1 - index_linear) * l_c_ + index_linear * l_c

    y_ = K.relu(y)

    if dc_decomp:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    else:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def minus(inputs_0, inputs_1, dc_decomp=False, grad_bounds=False, convex_domain={}):
    """

    :param inputs_0: tensor
    :param inputs_1: tensor
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :return: inputs_0 - inputs_1
    """

    if grad_bounds:
        raise NotImplementedError()

    if dc_decomp:
        y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_0

        y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_1

        h_ = h_0 - g_1
        g_ = g_0 - h_1

    else:
        y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0

        y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1

    w_u_ = w_u_0 - w_l_1
    w_l_ = w_l_0 - w_u_1

    b_u_ = b_u_0 - b_l_1
    b_l_ = b_l_0 - b_u_1

    u_c_ = u_c_0 - l_c_1
    l_c_ = l_c_0 - u_c_1

    upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
    u_c_ = K.minimum(upper_, u_c_)

    lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
    l_c_ = K.maximum(lower_, l_c_)

    y_ = y_0 - y_1

    if dc_decomp:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    else:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def add(inputs_0, inputs_1, dc_decomp=False, grad_bounds=False, convex_domain={}):
    """

    :param inputs_0: tensor
    :param inputs_1: tensor
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :return: inputs_0 + inputs_1
    """

    if grad_bounds:
        raise NotImplementedError()

    if dc_decomp:
        y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = inputs_0
        y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1, h_1, g_1 = inputs_1

        h_ = h_0 + h_1
        g_ = g_0 + g_1

    else:
        y_0, x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0
        y_1, _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1

    w_u_ = w_u_0 + w_u_1
    w_l_ = w_l_0 + w_l_1

    b_u_ = b_u_0 + b_u_1
    b_l_ = b_l_0 + b_l_1

    u_c_ = u_c_0 + u_c_1
    l_c_ = l_c_0 + l_c_1

    upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
    u_c_ = K.minimum(upper_, u_c_)

    lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
    l_c_ = K.maximum(lower_, l_c_)

    y_ = y_0 + y_1

    if dc_decomp:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    else:
        return [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


def maximum(inputs_0, inputs_1, dc_decomp=False, grad_bounds=False, convex_domain={}):
    """

    :param inputs_0: list of tensors
    :param inputs_1: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates
    whether we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :return: maximum(inputs_0, inputs_1)
    """

    return add(
        relu_(
            minus(
                inputs_1,
                inputs_0,
                dc_decomp=dc_decomp,
                grad_bounds=grad_bounds,
                convex_domain=convex_domain,
            ),
            dc_decomp,
            grad_bounds,
            convex_domain,
        ),
        inputs_0,
        dc_decomp,
        grad_bounds,
        convex_domain,
    )


# convex hull of the maximum between two functions
def max_(x, dc_decomp=False, grad_bounds=False, convex_domain={}, axis=-1):
    """

    :param x: list of tensors
    :param dc_decomp: boolean that indicates
    whether we return a difference of convex decomposition of our layer
    :param grad_bounds: boolean that indicates whether
    we propagate upper and lower bounds on the values of the gradient
    :param convex_domain: the type of convex domain
    :param axis: axis to perform the maximum
    :return: max operation  along an axis
    """

    if grad_bounds:
        raise NotImplementedError()

    if dc_decomp:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l, h, g = x
    else:
        y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

    input_shape = K.int_shape(y)
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    if dc_decomp:
        h_list = tf.split(h, max_dim, axis)
        g_list = tf.split(g, max_dim, axis)

    y_list = tf.split(y, max_dim, axis)
    b_u_list = tf.split(b_u, max_dim, axis)
    b_l_list = tf.split(b_l, max_dim, axis)

    u_c_list = tf.split(u_c, max_dim, axis)
    l_c_list = tf.split(l_c, max_dim, axis)

    if axis == -1:
        w_u_list = tf.split(w_u, max_dim, axis)
        w_l_list = tf.split(w_l, max_dim, axis)
    else:
        w_u_list = tf.split(w_u, max_dim, axis + 1)
        w_l_list = tf.split(w_l, max_dim, axis + 1)

    if dc_decomp:
        h_tmp = h_list[0] + K.zeros_like(h_list[0])
        g_tmp = g_list[0] + K.zeros_like(g_list[0])

    y_tmp = y_list[0] + K.zeros_like(y_list[0])
    b_u_tmp = b_u_list[0] + K.zeros_like(b_u_list[0])
    b_l_tmp = b_l_list[0] + K.zeros_like(b_l_list[0])
    u_c_tmp = u_c_list[0] + K.zeros_like(u_c_list[0])
    l_c_tmp = l_c_list[0] + K.zeros_like(l_c_list[0])
    w_u_tmp = w_u_list[0] + K.zeros_like(w_u_list[0])
    w_l_tmp = w_l_list[0] + K.zeros_like(w_l_list[0])

    if dc_decomp:
        output_tmp = [
            y_tmp,
            x_0,
            u_c_tmp,
            w_u_tmp,
            b_u_tmp,
            l_c_tmp,
            w_l_tmp,
            b_l_tmp,
            h_tmp,
            g_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [
                y_list[i],
                x_0,
                u_c_list[i],
                w_u_list[i],
                b_u_list[i],
                l_c_list[i],
                w_l_list[i],
                b_l_list[i],
                h_list[i],
                g_list[i],
            ]
            output_tmp = maximum(output_tmp, output_i, dc_decomp)
    else:
        output_tmp = [y_tmp, x_0, u_c_tmp, w_u_tmp, b_u_tmp, l_c_tmp, w_l_tmp, b_l_tmp]
        for i in range(1, max_dim):
            output_i = [
                y_list[i],
                x_0,
                u_c_list[i],
                w_u_list[i],
                b_u_list[i],
                l_c_list[i],
                w_l_list[i],
                b_l_list[i],
            ]
            output_tmp = maximum(output_tmp, output_i, dc_decomp)

    # reduce the dimension
    if dc_decomp:
        _, _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, _, _ = output_tmp
    else:
        _, _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_tmp

    b_u_ = K.squeeze(b_u_, axis)
    b_l_ = K.squeeze(b_l_, axis)
    u_c_ = K.squeeze(u_c_, axis)
    l_c_ = K.squeeze(l_c_, axis)

    if axis == -1:
        w_u_ = K.squeeze(w_u_, axis)
        w_l_ = K.squeeze(w_l_, axis)
    else:
        w_u_ = K.squeeze(w_u_, axis + 1)
        w_l_ = K.squeeze(w_l_, axis + 1)

    if dc_decomp:
        g_ = K.sum(g, axis=axis)
        h_ = K.max(h + g, axis=axis) - g_

    upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
    u_c_ = K.minimum(upper_, u_c_)

    lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
    l_c_ = K.maximum(lower_, l_c_)

    y_ = K.max(y, axis=axis)
    if dc_decomp:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
    else:
        output = [y_, x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    return output


def softmax_to_linear(model):
    layer = model.layers[-1]

    # check that layer is not an instance of the object Softmax
    if isinstance(layer, keras.layers.Softmax):
        model_normalize = keras.models.Model(model.get_input_at(0), keras.layers.Activation("linear")(layer.input))

        return model_normalize

    if hasattr(layer, "activation"):
        if not layer.get_config()["activation"] == "softmax":
            return model
        layer.get_config()["activation"] = "linear"
        layer.activation = keras.activations.get("linear")

    return model
