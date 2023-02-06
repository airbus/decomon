import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.math import greater_equal
from tensorflow.python.keras import backend as K

from decomon.layers.core import F_FORWARD, F_HYBRID, F_IBP, StaticVariables


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Grid:
    name = "grid"  # Hypercube
    stable_coeff = 0.0


class Vertex:
    name = "vertex"  # convex set represented by its vertices
    # (no verification is proceeded to assess that the set is convex)


# create static variables for varying convex domain
class V_slope:
    name = "volume-slope"


class A_slope:
    name = "adaptative-slope"


class S_slope:
    name = "same-slope"


class Z_slope:
    name = "zero-lb"


class O_slope:
    name = "one-lb"


# propagation of constant and affines bounds from input to output
class M_BACKWARD:
    name = "backward"


class M_FORWARD:
    name = "forward"


class M_REC_BACKWARD:
    name = "crown"


# linear hull for activation function
def relu_prime(x):
    """Derivative of relu

    Args:
        x

    Returns:

    """

    return K.clip(K.sign(x), K.cast(0, dtype=x.dtype), K.cast(1, dtype=x.dtype))


def sigmoid_prime(x):
    """Derivative of sigmoid

    Args:
        x

    Returns:

    """

    s_x = K.sigmoid(x)
    return s_x * (K.cast(1, dtype=x.dtype) - s_x)


def tanh_prime(x):
    """Derivative of tanh

    Args:
        x

    Returns:

    """

    s_x = K.tanh(x)
    return K.cast(1, dtype=x.dtype) - K.pow(s_x, K.cast(2, dtype=x.dtype))


def softsign_prime(x):
    """Derivative of softsign

    Args:
        x

    Returns:

    """

    return K.cast(1.0, dtype=x.dtype) / K.pow(K.cast(1.0, dtype=x.dtype) + K.abs(x), K.cast(2, dtype=x.dtype))


##############
# SYMBOLIC UPPER/ LOWER BOUNDS
# compute symbolically constant upper and lower
# with the current knowledge of the convex domain considered
##############

# case 1: a box
def get_upper_box(x_min, x_max, w, b, **kwargs):
    """#compute the max of an affine function
    within a box (hypercube) defined by its extremal corners

    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        max_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):  # identity function
        return x_max

    # split into positive and negative components
    z_value = K.cast(0.0, dtype=x_min.dtype)
    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_ = x_min + z_value * x_min
    x_max_ = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_max_ + w_neg * x_min_, 1) + b


def get_lower_box(x_min, x_max, w, b, **kwargs):
    """
    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w_l: weights of the affine lower bound
        b_l: bias of the affine lower bound

    Returns:
        min_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):
        return x_min

    z_value = K.cast(0.0, dtype=x_min.dtype)

    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_ = x_min + z_value * x_min
    x_max_ = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_ = K.expand_dims(x_min_, -1)
        x_max_ = K.expand_dims(x_max_, -1)

    return K.sum(w_pos * x_min_ + w_neg * x_max_, 1) + b


# case 2 : a ball
def get_lq_norm(x, p, axis=-1):
    """compute Lp norm (p=1 or 2)

    Args:
        x: tensor
        p: the power must be an integer in (1, 2)
        axis: the axis on which we compute the norm

    Returns:
        ||w||^p
    """
    if p == 1:
        x_q = K.max(K.abs(x), axis)
    elif p == 2:
        x_q = K.sqrt(K.sum(K.pow(x, p), axis))
    else:
        raise NotImplementedError("p must be equal to 1 or 2")

    return x_q


def get_upper_ball(x_0, eps, p, w, b, **kwargs):
    """max of an affine function over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        max_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        return x_0 + eps

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_upper_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        if len(kwargs):
            return get_upper_ball_finetune(x_0, eps, p, w, b, **kwargs)

        upper = eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + upper


def get_lower_ball(x_0, eps, p, w, b, **kwargs):
    """min of an affine fucntion over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        min_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        return x_0 - eps

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_lower_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        if len(kwargs):
            return get_lower_ball_finetune(x_0, eps, p, w, b, **kwargs)

        lower = -eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + lower


def get_lower_ball_finetune(x_0, eps, p, w, b, **kwargs):

    if "finetune_lower" in kwargs and "upper" in kwargs or "lower" in kwargs:

        alpha = kwargs["finetune_lower"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_ = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_ = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * lower_, 1) + K.sum(K.minimum(0.0, w_alpha_bar) * upper_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:

            upper = kwargs["upper"]  # flatten vector
            upper_ = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(0, w) * alpha[None] + K.maximum(0.0, w)
            w_alpha_bar = K.minimum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(0.0, w_alpha_bar) * upper_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:

            lower = kwargs["lower"]  # flatten vector
            lower_ = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(0, w) * alpha[None] + K.minimum(0.0, w)
            w_alpha_bar = K.maximum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * lower_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_lower_ball(x_0, eps, p, w, b)


def get_upper_ball_finetune(x_0, eps, p, w, b, **kwargs):

    if "finetune_upper" in kwargs and "upper" in kwargs or "lower" in kwargs:

        alpha = kwargs["finetune_upper"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_ = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_ = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * upper_, 1) + K.sum(K.minimum(0.0, w_alpha_bar) * lower_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:

            upper = kwargs["upper"]  # flatten vector
            upper_ = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(0, w) * alpha[None] + K.maximum(0.0, w)
            w_alpha_bar = K.minimum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * upper_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:

            lower = kwargs["lower"]  # flatten vector
            lower_ = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(0, w) * alpha[None] + K.minimum(0.0, w)
            w_alpha_bar = K.maximum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(0.0, w_alpha_bar) * lower_, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_upper_ball(x_0, eps, p, w, b)


def get_upper(x, w, b, convex_domain=None, **kwargs):
    """Meta function that aggregates all the way
    to compute a constant upper bounds depending on the convex domain

    Args:
        x: the tensors that represent the domain
        w: the weights of the affine function
        b: the bias
        convex_domain: the type of convex domain (see ???)

    Returns:
        a constant upper bound of the affine function
    """

    if convex_domain is None or len(convex_domain) == 0:
        # box
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b, **kwargs)

    elif convex_domain["name"] in {Box.name, Grid.name}:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min, x_max, w, b, **kwargs)

    elif convex_domain["name"] == Ball.name:

        eps = convex_domain["eps"]
        p = convex_domain["p"]
        return get_upper_ball(x, eps, p, w, b, **kwargs)

    elif convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    else:
        raise NotImplementedError()


def get_lower(x, w, b, convex_domain=None, **kwargs):
    """Meta function that aggregates all the way
    to compute a constant lower bound depending on the convex domain
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
        return get_lower_box(x_min, x_max, w, b, **kwargs)

    elif convex_domain["name"] in {Box.name, Grid.name}:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min, x_max, w, b, **kwargs)

    elif convex_domain["name"] == Ball.name:
        eps = convex_domain["eps"]
        p = convex_domain["p"]
        return get_lower_ball(x, eps, p, w, b, **kwargs)

    elif convex_domain["name"] == Vertex.name:
        raise NotImplementedError()

    else:
        raise NotImplementedError()


def get_lower_layer(convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    def func(inputs):
        return get_lower(inputs[0], inputs[1], inputs[2], convex_domain=convex_domain)

    return Lambda(func)


def get_upper_layer(convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    def func(inputs):
        return get_upper(inputs[0], inputs[1], inputs[2], convex_domain=convex_domain)

    return Lambda(func)


def get_lower_layer_box():
    def func(inputs):
        return get_lower_box(inputs[0], inputs[1], inputs[2], inputs[3])

    return Lambda(func)


def get_upper_layer_box():
    def func(inputs):
        return get_upper_box(inputs[0], inputs[1], inputs[2], inputs[3])

    return Lambda(func)


def backward_maximum(inputs_, convex_domain):

    back_bounds_0 = inputs_[2:6]
    back_bounds = inputs_[6:]

    output = inputs_[:2] + back_bounds_0
    for i in range(len(back_bounds) / 4):
        output = maximum(
            output,
            inputs_[:2] + back_bounds[4 * i : 4 * (i + 1)],
            dc_decomp=False,
            convex_domain=convex_domain,
            mode=F_FORWARD.name,
        )

    return output[-2:]


def backward_minimum(inputs_, convex_domain):

    back_bounds_0 = inputs_[2:6]
    back_bounds = inputs_[6:]

    output = inputs_[:2] + back_bounds_0
    for i in range(len(back_bounds) / 4):
        output = minimum(
            output,
            inputs_[:2] + back_bounds[4 * i : 4 * (i + 1)],
            dc_decomp=False,
            convex_domain=convex_domain,
            mode=F_FORWARD.name,
        )

    return output[2:4]


def noisy_lower(lower):

    # if some random binary variable is set to 0 return K.maximum(upper,- upper)
    var_ = K.minimum(lower, -lower)
    proba = K.random_binomial(lower.shape, p=0.2, dtype=K.floatx())

    return proba * lower + (1 - proba) * var_


def noisy_upper(upper):

    # if some random binary variable is set to 0 return K.maximum(upper,- upper)
    var_ = K.maximum(upper, -upper)
    proba = K.random_binomial(upper.shape, p=0.2, dtype=K.floatx())

    return proba * upper + (1 - proba) * var_


##### corners ######
def get_lower_bound_grid(x, W, b, n):

    A, B = convert_lower_search_2_subset_sum(x, W, b, n)
    return subset_sum_lower(A, B, repeat=n)


def get_upper_bound_grid(x, W, b, n):

    return -get_lower_bound_grid(x, -W, -b, n)


def get_bound_grid(x, W_u, b_u, W_l, b_l, n):

    upper = get_upper_bound_grid(x, W_u, b_u, n)
    lower = get_lower_bound_grid(x, W_l, b_l, n)

    return upper, lower


# convert max Wx +b s.t Wx+b<=0 into a subset-sum problem with positive values
def convert_lower_search_2_subset_sum(x, W, b, n):

    x_min = x[:, 0]
    x_max = x[:, 1]

    if len(W.shape) > 3:
        W = K.reshape(W, (-1, W.shape[1], np.prod(W.shape[2:])))
        b = K.reshape(b, (-1, np.prod(b.shape[1:])))

    const = get_lower(x, W, b, convex_domain={})

    weights = K.abs(W) * K.expand_dims((x_max - x_min) / n, -1)
    return weights, const


def subset_sum_lower(W, b, repeat=1):

    B = tf.sort(W, 1)
    C = K.repeat_elements(B, rep=repeat, axis=1)
    C_ = K.cumsum(C, axis=1)
    D = K.minimum(K.sign(K.expand_dims(-b, 1) - C_) + 1, 1)

    score = K.minimum(K.sum(D * C, 1) + b, 0.0)
    return score


# define routines to get linear relaxations useful both for forward and backward
def get_linear_hull_relu(upper, lower, slope, upper_g=0, lower_g=0, **kwargs):

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    alpha = (K.relu(upper) - K.relu(lower)) / K.maximum(K.cast(K.epsilon(), dtype=upper.dtype), upper - lower)

    # scaling factor for the upper bound on the relu
    # see README

    w_u_ = alpha
    b_u_ = K.relu(lower) - alpha * lower
    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == V_slope.name:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)

        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l_ = index_b
        b_l_ = z_value * b_u_

    if slope == A_slope.name:
        w_l_ = K.clip(K.sign(w_u_ - 0.5), 0, 1)
        b_l_ = z_value * b_u_

    if slope == Z_slope.name:
        w_l_ = z_value * w_u_
        b_l_ = z_value * b_u_

    if slope == O_slope.name:
        w_l_ = z_value * w_u_ + o_value
        b_l_ = z_value * b_u_

    if slope == S_slope.name:
        w_l_ = w_u_
        b_l_ = z_value * b_u_

    if "upper_grid" in kwargs:

        raise NotImplementedError()

    gamma = o_value
    if "finetune" in kwargs:
        # retrieve variables to optimize the slopes
        gamma = kwargs["finetune"][None]

    w_l_ = gamma * w_l_ + (o_value - gamma) * (o_value - w_l_)

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # =1 if inactive state
    index_linear = K.clip(K.sign(lower) + o_value, z_value, o_value)  # 1 if linear state

    w_u_ = (o_value - index_dead) * w_u_
    w_l_ = (o_value - index_dead) * w_l_
    b_u_ = (o_value - index_dead) * b_u_
    b_l_ = (o_value - index_dead) * b_l_

    w_u_ = (o_value - index_linear) * w_u_ + index_linear
    w_l_ = (o_value - index_linear) * w_l_ + index_linear
    b_u_ = (o_value - index_linear) * b_u_
    b_l_ = (o_value - index_linear) * b_l_

    return [w_u_, b_u_, w_l_, b_l_]


def get_linear_hull_sigmoid(upper, lower, slope, **kwargs):

    x = [upper, lower]
    return get_linear_hull_s_shape(
        x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain={}, mode=F_IBP.name, **kwargs
    )


def get_linear_hull_tanh(upper, lower, slope, **kwargs):

    x = [upper, lower]
    return get_linear_hull_s_shape(x, func=K.tanh, f_prime=tanh_prime, convex_domain={}, mode=F_IBP.name, **kwargs)


def get_linear_softplus_hull(upper, lower, slope, **kwargs):

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    u_c_ = K.softsign(upper)
    l_c_ = K.softsign(lower)
    alpha = (u_c_ - l_c_) / K.maximum(K.cast(K.epsilon(), dtype=upper.dtype), (upper - lower))
    w_u_ = alpha
    b_u_ = -alpha * lower + l_c_

    z_value = K.cast(0.0, dtype=upper.dtype)
    o_value = K.cast(1.0, dtype=upper.dtype)

    if slope == V_slope.name:
        # 1 if upper<=-lower else 0
        index_a = -K.clip(K.sign(upper + lower) - o_value, -o_value, z_value)
        # 1 if upper>-lower else 0
        index_b = o_value - index_a
        w_l_ = index_b
        b_l_ = z_value * b_u_
    elif slope == Z_slope.name:
        w_l_ = z_value * w_u_
        b_l_ = z_value * b_u_
    elif slope == O_slope.name:
        w_l_ = z_value * w_u_ + o_value
        b_l_ = z_value * b_u_
    elif slope == S_slope.name:
        w_l_ = w_u_
        b_l_ = z_value * b_u_
    else:
        raise ValueError(f"Unknown slope {slope}")

    if "finetune" in kwargs:
        # weighted linear combination
        alpha_l = kwargs["finetune"]
        alpha_l_ = alpha_l[None]

        w_l_ = alpha_l_ * w_l_
        b_l_ = alpha_l_ * b_l_ + (o_value - alpha_l_) * K.maximum(lower, z_value)

    index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)

    w_u_ = (o_value - index_dead) * w_u_
    w_l_ = (o_value - index_dead) * w_l_
    b_u_ = (o_value - index_dead) * b_u_
    b_l_ = (o_value - index_dead) * b_l_

    if "finetune" in kwargs:
        # weighted linear combination
        alpha_u, alpha_l = kwargs["finetune"]
        alpha_u_ = alpha_u[None]
        alpha_l_ = alpha_l[None]

        w_u_ = alpha_u_ * w_u_
        b_u_ = alpha_u_ * b_u_ + (o_value - alpha_u_) * K.maximum(upper, z_value)

        w_l_ = alpha_l_ * w_l_
        b_l_ = alpha_l_ * b_l_ + (o_value - alpha_l_) * K.maximum(lower, z_value)

    return [w_u_, b_u_, w_l_, b_l_]


def substract(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of inputs_0-inputs_1

    Args:
        inputs_0: tensor
        inputs_1: tensor
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        inputs_0 - inputs_1
    """
    if convex_domain is None:
        convex_domain = {}
    inputs_1_ = minus(inputs_1, mode=mode, dc_decomp=dc_decomp)
    output = add(inputs_0, inputs_1_, dc_decomp=dc_decomp, mode=mode, convex_domain=convex_domain)

    return output


def add(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of inputs_0+inputs_1

    Args:
        inputs_0: tensor
        inputs_1: tensor
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        inputs_0 + inputs_1
    """
    if convex_domain is None:
        convex_domain = {}
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if dc_decomp:
        h_0, g_0 = inputs_0[-2:]
        h_1, g_1 = inputs_1[-2:]
        h_ = h_0 + h_1
        g_ = g_0 + g_1

    if mode == F_HYBRID.name:
        x_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = inputs_0[:nb_tensor]
        _, u_c_1, w_u_1, b_u_1, l_c_1, w_l_1, b_l_1 = inputs_1[:nb_tensor]
    elif mode == F_IBP.name:
        u_c_0, l_c_0 = inputs_0[:nb_tensor]
        u_c_1, l_c_1 = inputs_1[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u_0, b_u_0, w_l_0, b_l_0 = inputs_0[:nb_tensor]
        _, w_u_1, b_u_1, w_l_1, b_l_1 = inputs_1[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [F_HYBRID.name, F_IBP.name]:
        u_c_ = u_c_0 + u_c_1
        l_c_ = l_c_0 + l_c_1
    if mode in [F_HYBRID.name, F_FORWARD.name]:

        w_u_ = w_u_0 + w_u_1
        w_l_ = w_l_0 + w_l_1

        b_u_ = b_u_0 + b_u_1
        b_l_ = b_l_0 + b_l_1

    if mode == F_HYBRID.name:
        upper_ = get_upper(x_0, w_u_, b_u_, convex_domain)
        u_c_ = K.minimum(upper_, u_c_)

        lower_ = get_lower(x_0, w_l_, b_l_, convex_domain)
        l_c_ = K.maximum(lower_, l_c_)

    if mode == F_HYBRID.name:
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    elif mode == F_IBP.name:
        output = [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output = [x_0, w_u_, b_u_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        output += [h_, g_]

    return output


def relu_(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, slope=V_slope.name, **kwargs):

    if convex_domain is None:
        convex_domain = {}
    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError(f"unknown mode {mode}")

    z_value = K.cast(0.0, dtype=x[0].dtype)
    o_value = K.cast(1.0, dtype=x[0].dtype)

    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensors]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    elif mode in [F_IBP.name, F_HYBRID.name]:
        upper = u_c
        lower = l_c
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        h, g = x[-2:]
        h_ = K.maximum(h, -g)
        g_ = g
        index_dead = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # =1 if inactive state
        index_linear = K.clip(K.sign(lower) + o_value, z_value, o_value)  # 1 if linear state

        h_ = (o_value - index_dead) * h_
        g_ = (o_value - index_dead) * g_
        h_ = (o_value - index_linear) * h_ + index_linear * h
        g_ = (o_value - index_linear) * g_ + index_linear * g

    u_c_ = K.relu(upper)
    l_c_ = K.relu(lower)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        if len(convex_domain) and convex_domain["name"] == Grid.name:
            upper_g, lower_g = get_bound_grid(x_0, w_u, b_u, w_l, b_l, 1)
            kwargs.update({"upper_grid": upper_g, "lower_grid": lower_g})

        w_u_, b_u_, w_l_, b_l_ = get_linear_hull_relu(upper, lower, slope, **kwargs)
        b_u_ = w_u_ * b_u + b_u_
        b_l_ = w_l_ * b_l + b_l_
        w_u_ = K.expand_dims(w_u_, 1) * w_u
        w_l_ = K.expand_dims(w_l_, 1) * w_l

    output = []
    if mode == F_IBP.name:
        output += [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output += [x_0, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        output += [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        output += [h_, g_]
    return output


def minus(inputs, mode=F_HYBRID.name, dc_decomp=False, **kwargs):
    """LiRPA implementation of minus(x)=-x.

    Args:
        inputs
        mode

    Returns:

    """
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        u, l = inputs[:nb_tensor]
    elif mode == F_FORWARD.name:
        x, w_u, b_u, w_l, b_l = inputs[:nb_tensor]
    elif mode == F_HYBRID.name:
        x, u, w_u, b_u, l, w_l, b_l = inputs[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        h, g = inputs[-2:]

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_ = -l
        l_ = -u

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_ = -w_l
        b_u_ = -b_l
        w_l_ = -w_u
        b_l_ = -b_u

    if mode == F_IBP.name:
        output = [u_, l_]
    elif mode == F_FORWARD.name:
        output = [x, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        output = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        output += [-g, -h]

    return output


def maximum(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, finetune=False, **kwargs):
    """LiRPA implementation of element-wise max

    Args:
        inputs_0: list of tensors
        inputs_1: list of tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
    whether we return a difference of convex decomposition of our layer

    Returns:
        maximum(inputs_0, inputs_1)
    """
    if convex_domain is None:
        convex_domain = {}
    output_0 = substract(inputs_1, inputs_0, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    if finetune:
        finetune = kwargs["finetune_params"]
        output_1 = relu_(output_0, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode, finetune=finetune)
    else:
        output_1 = relu_(output_0, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)

    return add(
        output_1,
        inputs_0,
        dc_decomp=dc_decomp,
        convex_domain=convex_domain,
        mode=mode,
    )


def minimum(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, finetune=False, **kwargs):
    """LiRPA implementation of element-wise min

    Args:
        inputs_0
        inputs_1
        dc_decomp
        convex_domain
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    return minus(
        maximum(
            minus(inputs_0, dc_decomp=dc_decomp, mode=mode),
            minus(inputs_1, dc_decomp=dc_decomp, mode=mode),
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode,
            finetune=finetune,
            **kwargs,
        ),
        dc_decomp=dc_decomp,
        mode=mode,
    )


def get_linear_hull_s_shape(x, func=K.sigmoid, f_prime=sigmoid_prime, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """Computing the linear hull of shape functions  given the pre activation neurons

    Args:
        x: list of input tensors
        func: the function (sigmoid, tanh, softsign...)
        f_prime: the derivative of the function (sigmoid_prime...)
        convex_domain: the type of convex input domain
        mode: type of Forward propagation (IBP, Forward or Hybrid)

    Returns:
        the updated list of tensors
    """

    if convex_domain is None:
        convex_domain = {}
    z_value = K.cast(0.0, dtype=x[0].dtype)
    o_value = K.cast(1.0, dtype=x[0].dtype)
    t_value = K.cast(2.0, dtype=x[0].dtype)

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        u_c, l_c = x[:nb_tensor]
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # flatten
    shape = list(u_c.shape[1:])
    u_c_flat = K.reshape(u_c, (-1, np.prod(shape)))  # (None, n)
    l_c_flat = K.reshape(l_c, (-1, np.prod(shape)))  # (None, n)

    # upper bound
    # derivative
    s_u_prime = f_prime(u_c_flat)  # (None, n)
    s_l_prime = f_prime(l_c_flat)  # (None, n)
    s_u = func(u_c_flat)  # (None, n)
    s_l = func(l_c_flat)  # (None, n)

    # case 0:
    coeff = (s_u - s_l) / K.maximum(K.cast(K.epsilon(), dtype=x[0].dtype), u_c_flat - l_c_flat)
    alpha_u_0 = K.switch(greater_equal(s_u_prime, coeff), o_value + z_value * u_c_flat, z_value * u_c_flat)  # (None, n)
    alpha_u_1 = (o_value - alpha_u_0) * ((K.sign(l_c_flat) + o_value) / t_value)

    w_u_0 = coeff
    b_u_0 = -w_u_0 * l_c_flat + s_l

    w_u_1 = z_value * u_c_flat
    b_u_1 = s_u

    w_u_2, b_u_2 = get_t_upper(u_c_flat, l_c_flat, s_l, func=func, f_prime=f_prime)

    w_u_ = K.reshape(alpha_u_0 * w_u_0 + alpha_u_1 * w_u_1 + (o_value - alpha_u_0 - alpha_u_1) * w_u_2, [-1] + shape)
    b_u_ = K.reshape(alpha_u_0 * b_u_0 + alpha_u_1 * b_u_1 + (o_value - alpha_u_0 - alpha_u_1) * b_u_2, [-1] + shape)

    # linear hull
    # case 0:
    alpha_l_0 = K.switch(greater_equal(s_l_prime, coeff), o_value + z_value * l_c_flat, z_value * l_c_flat)  # (None, n)
    alpha_l_1 = (o_value - alpha_l_0) * ((K.sign(-u_c_flat) + o_value) / t_value)

    w_l_0 = coeff
    b_l_0 = -w_l_0 * u_c_flat + s_u

    w_l_1 = z_value * u_c_flat
    b_l_1 = s_l

    w_l_2, b_l_2 = get_t_lower(u_c_flat, l_c_flat, s_u, func=func, f_prime=f_prime)

    w_l_ = K.reshape(alpha_l_0 * w_l_0 + alpha_l_1 * w_l_1 + (o_value - alpha_l_0 - alpha_l_1) * w_l_2, [-1] + shape)
    b_l_ = K.reshape(alpha_l_0 * b_l_0 + alpha_l_1 * b_l_1 + (o_value - alpha_l_0 - alpha_l_1) * b_l_2, [-1] + shape)

    return [w_u_, b_u_, w_l_, b_l_]


def get_t_upper(u_c_flat, l_c_flat, s_l, func=K.sigmoid, f_prime=sigmoid_prime):
    """linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine upper bound

    Args:
        u_c_flat: flatten tensor of constant upper bound
        l_c_flat: flatten tensor of constant lower bound
        s_l: lowest value of the function func on the domain
        func: the function (sigmoid, tanh,  softsign)
        f_prime: the derivative of the function

    Returns:
        the upper affine bounds in this subcase
    """

    o_value = K.cast(1.0, dtype=u_c_flat.dtype)
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = K.cast(np.linspace(0, 1, 100)[None, None, :], dtype=u_c_flat.dtype) * u_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (s_t_ - K.expand_dims(s_l, -1)) / (t_ - l_c_flat_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)
    threshold = K.min(score, -1)  # (None, n)

    index_t = K.cast(
        K.switch(K.greater(threshold, z_value * threshold), index_, K.clip(index_ - 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], dtype=u_c_flat.dtype) + z_value * u_c_flat_,
                K.expand_dims(index_t, -1) + z_value * u_c_flat_,
            ),
            t_,
            z_value * t_,
        ),
        -1,
    )  # (None, n)

    s_t = func(t_value)  # (None, n)
    w_u = (s_t - s_l) / K.maximum(K.cast(K.epsilon(), dtype=u_c_flat.dtype), t_value - l_c_flat)  # (None, n)
    b_u = -w_u * l_c_flat + s_l  # + func(l_c_flat)

    return [w_u, b_u]


def get_t_lower(u_c_flat, l_c_flat, s_u, func=K.sigmoid, f_prime=sigmoid_prime):
    """linear interpolation between lower and upper bounds on the function func to have a symbolic approximation of the best
    coefficient for the affine lower bound

    Args:
        u_c_flat: flatten tensor of constant upper bound
        l_c_flat: flatten tensor of constant lower bound
        s_u: highest value of the function func on the domain
        func: the function (sigmoid, tanh,  softsign)
        f_prime: the derivative of the function

    Returns:
        the lower affine bounds in this subcase
    """
    z_value = K.cast(0.0, dtype=u_c_flat.dtype)
    o_value = K.cast(1.0, dtype=u_c_flat.dtype)

    # step1: find t
    u_c_flat_ = K.expand_dims(u_c_flat, -1)  # (None, n , 1)
    l_c_flat_ = K.expand_dims(l_c_flat, -1)  # (None, n,  1)
    t_ = K.cast(np.linspace(0, 1.0, 100)[None, None, :], dtype=u_c_flat.dtype) * l_c_flat_  # (None, n , 100)

    s_p_t_ = f_prime(t_)  # (None, n, 100)
    s_t_ = func(t_)  # (None, n, 100)

    score = K.abs(s_p_t_ - (K.expand_dims(s_u, -1) - s_t_) / (u_c_flat_ - t_))  # (None, n, 100)
    index_ = K.argmin(score, -1)  # (None, n)

    threshold = K.min(score, -1)
    index_t = K.cast(
        K.switch(K.greater(threshold, z_value * threshold), index_, K.clip(index_ + 1, 0, 100)), dtype=u_c_flat.dtype
    )  # (None, n)
    t_value = K.sum(
        K.switch(
            K.equal(
                o_value * K.cast(np.arange(0, 100)[None, None, :], dtype=u_c_flat.dtype) + z_value * u_c_flat_,
                K.expand_dims(index_t, -1) + z_value * u_c_flat_,
            ),
            t_,
            z_value * t_,
        ),
        -1,
    )

    s_t = func(t_value)  # (None, n)
    w_l = (s_u - s_t) / K.maximum(K.cast(K.epsilon(), dtype=u_c_flat.dtype), u_c_flat - t_value)  # (None, n)
    b_l = -w_l * u_c_flat + s_u  # func(u_c_flat)

    return [w_l, b_l]


def set_mode(x, final_mode, mode, convex_domain=None):
    if convex_domain is None:
        convex_domain = {}

    if final_mode == mode:
        return x

    x_0, u_c, w_u, b_u, l_c, w_l, b_l = None, None, None, None, None, None, None
    if mode == F_IBP.name:
        u_c, l_c = x
        if final_mode != mode:
            raise NotImplementedError(f"If mode if {F_IBP}, final_mode must be also {F_IBP}.")
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x
        if final_mode in [F_IBP.name, F_HYBRID.name]:
            # compute constant bounds
            u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
    else:
        raise ValueError(f"Unknown mode {mode}")

    if final_mode == F_IBP.name:
        return [u_c, l_c]
    elif final_mode == F_FORWARD.name:
        return [x_0, w_u, b_u, w_l, b_l]
    elif final_mode == F_HYBRID.name:
        return [x_0, u_c, w_u, b_u, l_c, w_l, b_l]
    else:
        raise ValueError(f"Unknown final_mode {final_mode}")


def get_AB(model_):
    dico_AB = {}
    convex_domain = model_.convex_domain
    if not (len(convex_domain) and convex_domain["name"] == "grid" and convex_domain["option"] == "milp"):
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.grid_finetune
    return dico_AB


def get_AB_finetune(model_):
    dico_AB = {}
    convex_domain = model_.convex_domain
    if not (len(convex_domain) and convex_domain["name"] == "grid" and convex_domain["option"] == "milp"):
        return dico_AB

    if not model_.finetune:
        return dico_AB

    for layer in model_.layers:
        name = layer.name
        sub_names = name.split("backward_activation")
        if len(sub_names) > 1:
            key = f"{layer.layer.name}_{layer.rec}"
            if key not in dico_AB:
                dico_AB[key] = layer.alpha_b_l
    return dico_AB