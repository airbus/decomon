import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Initializer
from tensorflow.python.keras import backend as K

from decomon.layers.core import F_FORWARD, F_HYBRID, F_IBP, Ball, Box, StaticVariables
from decomon.utils import (
    V_slope,
    add,
    get_linear_softplus_hull,
    get_lower,
    get_upper,
    maximum,
    minimum,
    minus,
    relu_,
)


#####
# USE SYMBOLIC GRADIENT DESCENT WITH OVERESTIMATION GUARANTEES
#####
# first compute gradient of the function
def get_grad(x, constant, W, b):
    """We compute the gradient of the function f at sample x

    f = sum_{i<= n_linear} max(constant_i, W_i*x + b_i)
    it is quite easy to compute the gradient symbolically, without using the gradient operator
    it is either 0 if f(x) = constant or W else
    this trick allows to be backpropagation compatible

    Args:
        x: Keras Tensor (None, n_dim, ...)
        constant: Keras Tensor (None, n_linear, ...)
        W: Keras Tensor (None, n_dim, n_linear, ...)
        b: Keras Tensor (None, n_linear, ...)

    Returns:
        Keras Tensor the gradient of the function (None, n_dim, ...)
    """
    # product W*x + b
    x_ = K.expand_dims(x, 2)
    z = K.sum(W * x_, 1) + b
    grad_ = K.sum(K.expand_dims(-K.sign(constant - K.maximum(constant, z)), 1) * W, 2)  # (None, n_dim, ...)

    return grad_


def compute_L(W):
    """We compute the largest possible norm of the gradient

    Args:
        W: Keras Tensor (None, n_dim, n_linear, ...)

    Returns:
        Keras Tensor with an upper bound on the largest magnitude of the
        gradient
    """
    return K.sum(K.sqrt(K.sum(W * W, 1)), 1)


def compute_R(z, convex_domain):
    """We compute the largest L2 distance of the starting point with the global optimum

    Args:
        z: Keras Tensor
        convex_domain: Dictionnary to complement z on the convex domain

    Returns:
        Keras Tensor an upper bound on the distance
    """

    if len(convex_domain) == 0:
        # compute the L2 distance z[:, 0], z[:, 1]
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Box.name:
        dist_ = K.sqrt(K.sum(K.pow((z[:, 1] - z[:, 0]) / 2.0, 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        dist_ = K.sqrt(K.sum(K.pow(z - z + convex_domain["eps"], 2), -1))
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        dist_ = convex_domain["eps"] * (0 * z + 1.0)
    else:
        raise NotImplementedError()

    return dist_


def get_start_point(z, convex_domain):
    """Create a warm start for the optimization (the mid point to minimize the largest distance between the
    warm start and the global optimum

    Args:
        z: Keras Tensor
        convex_domain: Dictionnary to complement z on the convex domain

    Returns:
        Keras Tensor
    """

    if len(convex_domain) == 0:
        return z[:, 0]
    elif convex_domain["name"] == Box.name:
        return (z[:, 0] + z[:, 1]) / 2.0
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == np.inf:
        return z
    elif convex_domain["name"] == Ball.name and convex_domain["p"] == 2:
        return z
    else:
        raise NotImplementedError()


def get_coeff_grad(R, k, g):
    """
    Args:
        R: Keras Tensor that reprends the largest distance to the gloabl
            optimum
        k: the number of iteration done so far
        g: the gradient

    Returns:
        the adaptative step size for the gradient
    """

    denum = np.sqrt(k) * K.sqrt(K.sum(K.pow(g, 2), 1))

    alpha = R / K.maximum(K.epsilon(), denum)
    return alpha


def grad_descent_conv(z, concave_upper, convex_lower, op_pos, ops_neg, n_iter):
    """
    Args:
        z
        concave_upper
        convex_lower
        op_pos
        ops_neg
        n_iter

    Returns:

    """

    raise NotImplementedError()


def grad_descent(z, convex_0, convex_1, convex_domain, n_iter=5):
    """
    Args:
        z: Keras Tensor
        constant: Keras Tensor, the constant of each component
        W: Keras Tensor, the affine of each component
        b: Keras Tensor, the bias of each component
        convex_domain: Dictionnary to complement z on the convex domain
        n_iter: the number of total iteration

    Returns:

    """

    constant_0, W_0, b_0 = convex_0
    constant_1, W_1, b_1 = convex_1

    # init
    x_k = K.expand_dims(get_start_point(z, convex_domain), -1) + 0 * K.sum(constant_0, 1)[:, None]
    R = compute_R(z, convex_domain)
    n_dim = len(x_k.shape[1:])
    while n_dim > 1:
        R = K.expand_dims(R, -1)
        n_dim -= 1

    def step_grad(x_, x_k_):

        x_k = x_k_[0]
        g_k_0 = get_grad(x_k, constant_0, W_0, b_0)
        g_k_1 = get_grad(x_k, constant_1, W_1, b_1)
        g_k = g_k_0 + g_k_1
        alpha_k = get_coeff_grad(R, n_iter + 1, g_k)
        x_result = alpha_k[:, None] * g_k
        return x_result, [x_k]

    x_vec = K.rnn(
        step_function=step_grad, inputs=K.concatenate([x_k[:, None]] * n_iter, 1), initial_states=[x_k], unroll=False
    )[1]

    # check convergence
    x_k = x_vec[:, -1]
    g_k = get_grad(x_k, constant_0, W_0, b_0) + get_grad(x_k, constant_1, W_1, b_1)
    mask_grad = K.sign(K.sqrt(K.sum(K.pow(g_k, 2), 1)))  # check whether we have converge
    X_vec = K.expand_dims(x_vec, -2)
    f_vec = K.sum(
        K.maximum(constant_0[:, None], K.sum(W_0[:, None] * X_vec, 2) + b_0[:, None])
        + K.maximum(constant_1[:, None], K.sum(W_1[:, None] * X_vec, 2) + b_1[:, None]),
        2,
    )
    f_vec = f_vec[0]
    L_0 = compute_L(W_0)
    L_1 = compute_L(W_1)
    L = L_0 + L_1
    penalty = (L * R) / np.sqrt(n_iter + 1)
    return f_vec - mask_grad * penalty


class NonPos(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w):
        return K.minimum(w, 0.0)


class NonNeg(Constraint):
    """Constrains the weights to be non-negative."""

    def __call__(self, w):
        return K.maximum(w, 0.0)


class ClipAlpha(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w):
        return K.clip(w, 0.0, 1.0)


class ClipAlphaGrid(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w):
        w = K.clip(w, 0.0, 1.0)
        w /= K.maximum(K.sum(w, 0), 1.0)[None]
        return w


class ClipAlphaAndSumtoOne(Constraint):
    """Cosntraints the weights to be between 0 and 1."""

    def __call__(self, w):
        w = K.clip(w, 0.0, 1.0)
        # normalize the first colum to 1
        w_scale = K.maximum(K.sum(w, 0), K.epsilon())
        return w / w_scale[:, None, None, None]


class MultipleConstraint(Constraint):
    """stacking multiple constraints"""

    def __init__(self, constraint_0, constraint_1, **kwargs):
        super().__init__(**kwargs)
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
        super().__init__(**kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        w_ = self.initializer.__call__(shape, dtype)
        return K.maximum(0.0, w_)


class Project_initializer_neg(Initializer):
    """Initializer that generates tensors initialized to 1."""

    def __init__(self, initializer, **kwargs):
        super().__init__(**kwargs)
        self.initializer = initializer

    def __call__(self, shape, dtype=None, **kwargs):
        w_ = self.initializer.__call__(shape, dtype)
        return K.minimum(0.0, w_)


def softplus_(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, slope=V_slope.name, **kwargs):

    if convex_domain is None:
        convex_domain = {}
    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError(f"unknown mode {mode}")

    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensors]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if dc_decomp:
        h, g = x[-2:]

    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    elif mode == F_HYBRID.name:
        upper = u_c
        lower = l_c
    elif mode == F_IBP.name:
        upper = u_c
        lower = l_c
    else:
        raise ValueError(f"Unknown mode {mode}")

    u_c_ = K.softplus(upper)
    l_c_ = K.softplus(lower)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        w_u_, b_u_, w_l_, b_l_ = get_linear_softplus_hull(upper, lower, slope, **kwargs)
        b_u_ = w_u_ * b_u + b_u_
        b_l_ = w_l_ * b_l + b_l_
        w_u_ = K.expand_dims(w_u_, 1) * w_u
        w_l_ = K.expand_dims(w_l_, 1) * w_l

    if mode == F_IBP.name:
        return [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        return [x_0, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")


def sum(x, axis=-1, dc_decomp=False, mode=F_HYBRID.name, **kwargs):

    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        return [K.sum(x[0], axis=axis), K.sum(x[1], axis=axis)]
    elif mode in [F_FORWARD.name, F_HYBRID.name]:
        if mode == F_FORWARD.name:
            x_0, w_u, b_u, w_l, b_l = x
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
            u_c_ = K.sum(u_c, axis=axis)
            l_c_ = K.sum(l_c, axis=axis)

        axis_w = -1
        if axis != -1:
            axis_w = axis + 1
        w_u_ = K.sum(w_u, axis=axis_w)
        w_l_ = K.sum(w_l, axis=axis_w)
        b_u_ = K.sum(b_u, axis=axis)
        b_l_ = K.sum(b_l, axis=axis)

        if mode == F_FORWARD.name:
            return [x_0, w_u_, b_u_, w_l_, b_l_]
        else:
            return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")


def frac_pos(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    # frac_pos is convex for positive values
    if mode == F_IBP.name:
        u_c, l_c = x
        u_c_ = 1.0 / l_c
        l_c_ = 1.0 / u_c
        return [u_c_, l_c_]
    if mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x
        u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x
        u_c_ = 1.0 / l_c
        l_c_ = 1.0 / u_c

        w_u_0 = (u_c_ - l_c_) / K.maximum(u_c - l_c, K.epsilon())
        b_u_0 = l_c_ - w_u_0 * l_c

        y = (u_c + l_c) / 2.0
        b_l_0 = 2.0 / y
        w_l_0 = -1 / y**2

        w_u_ = w_u_0[:, None] * w_l
        b_u_ = b_u_0 * b_l + b_u_0
        w_l_ = w_l_0[:, None] * w_u
        b_l_ = b_l_0 * b_u + b_l_0

        if mode == F_FORWARD.name:
            return [x_0, w_u_, b_u_, w_l_, b_l_]
        if mode == F_HYBRID.name:
            return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]


# convex hull of the maximum between two functions
def max_(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, axis=-1, finetune=False, **kwargs):
    """LiRPA implementation of max(x, axis)

    Args:
        x: list of tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
        axis: axis to perform the maximum
    whether we return a difference of convex decomposition of our layer

    Returns:
        max operation  along an axis
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        h, g = x[-2:]

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode == F_IBP.name and not dc_decomp:
        u_c_ = K.max(u_c, axis=axis)
        l_c_ = K.max(l_c, axis=axis)

        return [u_c_, l_c_]

    input_shape = K.int_shape(x[-1])
    max_dim = input_shape[axis]

    # do some transpose so that the last axis is also at the end

    if dc_decomp:
        h_list = tf.split(h, max_dim, axis)
        g_list = tf.split(g, max_dim, axis)

    if mode in [F_HYBRID.name, F_IBP.name]:

        u_c_list = tf.split(u_c, max_dim, axis)
        l_c_list = tf.split(l_c, max_dim, axis)
        u_c_tmp = u_c_list[0] + 0 * (u_c_list[0])
        l_c_tmp = l_c_list[0] + 0 * (l_c_list[0])

    if mode in [F_HYBRID.name, F_FORWARD.name]:

        b_u_list = tf.split(b_u, max_dim, axis)
        b_l_list = tf.split(b_l, max_dim, axis)
        b_u_tmp = b_u_list[0] + 0 * (b_u_list[0])
        b_l_tmp = b_l_list[0] + 0 * (b_l_list[0])

        if axis == -1:
            w_u_list = tf.split(w_u, max_dim, axis)
            w_l_list = tf.split(w_l, max_dim, axis)
        else:
            w_u_list = tf.split(w_u, max_dim, axis + 1)
            w_l_list = tf.split(w_l, max_dim, axis + 1)
        w_u_tmp = w_u_list[0] + 0 * (w_u_list[0])
        w_l_tmp = w_l_list[0] + 0 * (w_l_list[0])

        if finetune:
            key = [e for e in kwargs.keys()][0]
            params = kwargs[key][0]
            params_ = [e[0] for e in tf.split(params[None], max_dim, axis)]

    output_tmp = []
    if mode == F_HYBRID.name:
        output_tmp = [
            x_0,
            u_c_tmp,
            w_u_tmp,
            b_u_tmp,
            l_c_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(
                    output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_[i]
                )
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)

    elif mode == F_IBP.name:
        output_tmp = [
            u_c_tmp,
            l_c_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [u_c_list[i], l_c_list[i]]
            output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune)

    elif mode == F_FORWARD.name:
        output_tmp = [
            x_0,
            w_u_tmp,
            b_u_tmp,
            w_l_tmp,
            b_l_tmp,
        ]

        for i in range(1, max_dim):
            output_i = [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
            if finetune:
                output_tmp = maximum(
                    output_tmp, output_i, dc_decomp=False, mode=mode, finetune=finetune, finetune_params=params_[i]
                )
            else:
                output_tmp = maximum(output_tmp, output_i, dc_decomp=False, mode=mode)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # reduce the dimension
    if mode == F_IBP.name:
        u_c_, l_c_ = output_tmp[:nb_tensor]
    elif mode == F_FORWARD.name:
        _, w_u_, b_u_, w_l_, b_l_ = output_tmp[:nb_tensor]
    elif mode == F_HYBRID.name:
        _, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output_tmp[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = K.squeeze(u_c_, axis)
        l_c_ = K.squeeze(l_c_, axis)
    if mode in [F_HYBRID.name, F_FORWARD.name]:
        b_u_ = K.squeeze(b_u_, axis)
        b_l_ = K.squeeze(b_l_, axis)
        if axis == -1:
            w_u_ = K.squeeze(w_u_, axis)
            w_l_ = K.squeeze(w_l_, axis)
        else:
            w_u_ = K.squeeze(w_u_, axis + 1)
            w_l_ = K.squeeze(w_l_, axis + 1)

    if dc_decomp:
        g_ = K.sum(g, axis=axis)
        h_ = K.max(h + g, axis=axis) - g_

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


def softmax_to_linear(model):
    """linearize the softmax layer for verification

    Args:
        model: Keras Model

    Returns:
        model without the softmax
    """
    layer = model.layers[-1]
    # check that layer is not an instance of the object Softmax
    if isinstance(layer, keras.layers.Softmax):
        model_normalize = keras.models.Model(model.get_input_at(0), keras.layers.Activation("linear")(layer.input))

        return model_normalize, True

    if hasattr(layer, "activation"):
        if not layer.get_config()["activation"] == "softmax":
            return model, False
        layer.get_config()["activation"] = "linear"
        layer.activation = keras.activations.get("linear")
        return model, True

    return model, False


def linear_to_softmax(model):

    model.layers[-1].activation = keras.activations.get("softmax")
    return model


def multiply_old(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of (element-wise) multiply(x,y)=-x*y.

    Args:
        inputs_0: list of tensors
        inputs_1: list of tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
        mode: type of Forward propagation (IBP, Forward or Hybrid)
    whether we return a difference of convex decomposition of our layer
    whether we propagate upper and lower bounds on the values of the gradient

    Returns:
        maximum(inputs_0, inputs_1)
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    nb_tensor = StaticVariables(dc_decomp, mode=mode).nb_tensors

    if mode == F_IBP.name:
        u0, l0 = inputs_0[:nb_tensor]
        u1, l1 = inputs_1[:nb_tensor]
    elif mode == F_FORWARD.name:
        x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:nb_tensor]
        u0 = get_upper(x, w_u0, b_u0, convex_domain=convex_domain)
        l0 = get_lower(x, w_l0, b_l0, convex_domain=convex_domain)
        u1 = get_upper(x, w_u1, b_u1, convex_domain=convex_domain)
        l1 = get_lower(x, w_l1, b_l1, convex_domain=convex_domain)
    elif mode == F_HYBRID.name:
        x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")
    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U
    if mode in [F_IBP.name, F_HYBRID.name]:

        upper_0 = K.relu(u0) * u1 - K.relu(-u0) * l1 + K.relu(l1) * u0 - K.relu(-l1) * l0 - u0 * l1
        upper_1 = K.relu(u1) * u0 - K.relu(-u1) * l0 + K.relu(l0) * u1 - K.relu(-l0) * l1 - u1 * l0

        lower_0 = K.relu(l0) * l1 - K.relu(-l0) * u1 + K.relu(l1) * l0 - K.relu(-l1) * u0 - l0 * l1
        lower_1 = K.relu(u1) * l0 - K.relu(-u1) * u0 + K.relu(u0) * l1 - K.relu(-u0) * u1 - u1 * u0

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_0_u = (
            K.relu(u0)[:, None] * w_u1
            - K.relu(-u0)[:, None] * w_l1
            + K.relu(l1)[:, None] * w_u0
            - K.relu(-l1)[:, None] * w_l0
        )
        w_1_u = (
            K.relu(u1)[:, None] * w_u0
            - K.relu(-u1)[:, None] * w_l0
            + K.relu(l0)[:, None] * w_u1
            - K.relu(-l0)[:, None] * w_l1
        )

        w_0_l = (
            K.relu(l0)[:, None] * w_l1
            - K.relu(-l0)[:, None] * w_u1
            + K.relu(l1)[:, None] * w_l0
            - K.relu(-l1)[:, None] * w_u0
        )
        w_1_l = (
            K.relu(u1)[:, None] * w_l0
            - K.relu(-u1)[:, None] * w_u0
            + K.relu(u0)[:, None] * w_l1
            - K.relu(-u0)[:, None] * w_u1
        )

        b_u_0 = K.relu(u0) * b_u1 - K.relu(-u0) * b_l1 + K.relu(l1) * b_u0 - K.relu(-l1) * b_l0 - u0 * l1
        b_u_1 = K.relu(u1) * b_u0 - K.relu(-u1) * b_l0 + K.relu(l0) * b_u1 - K.relu(-l0) * b_l1 - u1 * l0

        b_l_0 = K.relu(l0) * b_l1 - K.relu(-l0) * b_u1 + K.relu(l1) * b_l0 - K.relu(-l1) * b_u0 - l0 * l1
        b_l_1 = K.relu(u1) * b_l0 - K.relu(-u1) * b_u0 + K.relu(u0) * b_l1 - K.relu(-u0) * b_u1 - u1 * u0

    if mode == F_IBP.name:
        output = [K.minimum(upper_0, upper_1), K.maximum(lower_0, lower_1)]
    elif mode in [F_HYBRID.name, F_FORWARD.name]:
        if mode == F_FORWARD.name:
            inputs_0_ = [x, w_0_u, b_u_0, w_0_l, b_l_0]
            inputs_1_ = [x, w_1_u, b_u_1, w_1_l, b_l_1]
        else:
            inputs_0_ = [x, upper_0, w_0_u, b_u_0, lower_0, w_0_l, b_l_0]
            inputs_1_ = [x, upper_1, w_1_u, b_u_1, lower_1, w_1_l, b_l_1]

        output_upper = minimum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
        output_lower = maximum(inputs_0_, inputs_1_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)

        if mode == F_FORWARD.name:
            _, w_u_, b_u_, _, _ = output_upper
            _, _, _, w_l_, b_l_ = output_lower
            output = [x, w_u_, b_u_, w_l_, b_l_]
        else:
            _, u_, w_u_, b_u_, _, _, _ = output_upper
            _, _, _, _, l_, w_l_, b_l_ = output_upper
            output = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return output


def multiply(inputs_0, inputs_1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of (element-wise) multiply(x,y)=-x*y.

    Args:
        inputs_0: list of tensors
        inputs_1: list of tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex domain
        mode: type of Forward propagation (IBP, Forward or Hybrid)
    whether we return a difference of convex decomposition of our layer
    whether we propagate upper and lower bounds on the values of the gradient

    Returns:
        maximum(inputs_0, inputs_1)
    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    nb_tensor = StaticVariables(dc_decomp, mode=mode).nb_tensors

    if mode == F_IBP.name:
        u0, l0 = inputs_0[:nb_tensor]
        u1, l1 = inputs_1[:nb_tensor]
    elif mode == F_FORWARD.name:
        x, w_u0, b_u0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, w_u1, b_u1, w_l1, b_l1 = inputs_1[:nb_tensor]
        u0 = get_upper(x, w_u0, b_u0, convex_domain=convex_domain)
        l0 = get_lower(x, w_l0, b_l0, convex_domain=convex_domain)
        u1 = get_upper(x, w_u1, b_u1, convex_domain=convex_domain)
        l1 = get_lower(x, w_l1, b_l1, convex_domain=convex_domain)
    elif mode == F_HYBRID.name:
        x, u0, w_u0, b_u0, l0, w_l0, b_l0 = inputs_0[:nb_tensor]
        _, u1, w_u1, b_u1, l1, w_l1, b_l1 = inputs_1[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")
    # using McCormick's inequalities to derive bounds
    # xy<= x_u*y + x*y_L - xU*y_L
    # xy<= x*y_u + x_L*y - x_L*y_U

    # xy >=x_L*y + x*y_L -x_L*y_L
    # xy >= x_U*y + x*y_U - x_U*y_U

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_0_ = K.maximum(u0 * u1, u0 * l1) + K.maximum(u0 * l1, l0 * l1) - u0 * l1
        u_c_1_ = K.maximum(u1 * u0, u1 * l0) + K.maximum(u1 * l0, l1 * l0) - u1 * l0
        u_c_ = K.minimum(u_c_0_, u_c_1_)
        l_c_ = K.minimum(l0 * l1, l0 * u1) + K.minimum(l0 * l1, u0 * l1) - l0 * l1

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        # xy <= x_u * y + x * y_L - xU * y_L
        cx_u_pos = K.maximum(u0, 0.0)
        cx_u_neg = K.minimum(u0, 0.0)

        cy_l_pos = K.maximum(l1, 0.0)
        cy_l_neg = K.minimum(l1, 0.0)
        w_u_ = cx_u_pos[:, None] * w_u1 + cx_u_neg[:, None] * w_l1 + cy_l_pos[:, None] * w_u0 + cy_l_neg[:, None] * w_l0
        b_u_ = cx_u_pos * b_u1 + cx_u_neg * b_l1 + cy_l_pos * b_u0 + cy_l_neg * b_l0 - u0 * l1

        # xy >= x_U*y + x*y_U - x_U*y_U
        cx_l_pos = K.maximum(l0, 0.0)
        cx_l_neg = K.minimum(l0, 0.0)

        w_l_ = cx_l_pos[:, None] * w_l1 + cx_l_neg[:, None] * w_u1 + cy_l_pos[:, None] * w_l0 + cy_l_neg[:, None] * w_u0
        b_l_ = cx_l_pos * b_l1 + cx_l_neg * b_u1 + cy_l_pos * b_l0 + cy_l_neg * b_u0 - l0 * l1

    if mode == F_IBP.name:
        return [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        return [x, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        return [x, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")


def permute_dimensions(x, axis, mode=F_HYBRID.name, axis_perm=1):
    """LiRPA implementation of (element-wise) permute(x,axis)

    Args:
        x: list of input tensors
        axis: axis on which we apply the permutation
        mode: type of Forward propagation (IBP, Forward or Hybrid)
        axis_perm: see DecomonPermute operator

    Returns:

    """

    if len(x[0].shape) <= 2:
        return x
    index = np.arange(len(x[0].shape))
    index = np.insert(np.delete(index, axis), axis_perm, axis)
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        return [
            K.permute_dimensions(x[2], index),
            K.permute_dimensions(x[3], index),
        ]
    else:
        index_w = np.arange(len(x[0].shape) + 1)
        index_w = np.insert(np.delete(index_w, axis), axis_perm + 1, axis)

        if mode == F_FORWARD.name:
            x_0, w_u, b_u, w_l, b_l = x[:nb_tensor]
            return [
                x_0,
                K.permute_dimensions(w_u, index_w),
                K.permute_dimensions(b_u, index),
                K.permute_dimensions(w_l, index_w),
                K.permute_dimensions(b_l, index),
            ]

        elif mode == F_HYBRID.name:
            x_0, u, w_u, b_u, l, w_l, b_l = x

            return [
                x_0,
                K.permute_dimensions(u, index),
                K.permute_dimensions(w_u, index_w),
                K.permute_dimensions(b_u, index),
                K.permute_dimensions(l, index),
                K.permute_dimensions(w_l, index_w),
                K.permute_dimensions(b_l, index),
            ]

        else:
            raise ValueError(f"Unknown mode {mode}")


def broadcast(inputs, n, axis, mode):
    """LiRPA implementation of broadcasting

    Args:
        inputs
        n
        axis
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

    if mode in [F_IBP.name, F_HYBRID.name]:
        for _ in range(n):
            u = K.expand_dims(u, axis)
            l = K.expand_dims(l, axis)

    if axis != -1:
        axis_w = axis + 1
    else:
        axis_w = -1

    if mode in [F_FORWARD.name, F_HYBRID.name]:
        for _ in range(n):
            b_u = K.expand_dims(b_u, axis)
            b_l = K.expand_dims(b_l, axis)
            w_u = K.expand_dims(w_u, axis_w)
            w_l = K.expand_dims(w_l, axis_w)

    if mode == F_IBP.name:
        output = [u, l]
    elif mode == F_FORWARD.name:
        output = [x, w_u, b_u, w_l, b_l]
    elif mode == F_HYBRID.name:
        output = [x, u, w_u, b_u, l, w_l, b_l]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return output


def split(input_, axis=-1, mode=F_HYBRID.name):
    """LiRPA implementation of split

    Args:
        input_
        axis
        mode

    Returns:

    """

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        u_, l_ = input_[:nb_tensor]
    elif mode == F_HYBRID.name:
        x_, u_, w_u_, b_u_, l_, w_l_, b_l_ = input_[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_, w_u_, b_u_, w_l_, b_l_ = input_[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_list = tf.split(u_, 1, axis=axis)
        l_list = tf.split(l_, 1, axis=axis)
        n = len(u_list)

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        b_u_list = tf.split(b_u_, 1, axis=axis)
        b_l_list = tf.split(b_l_, 1, axis=axis)
        n = len(b_u_list)

        if axis != -1:
            axis += 1
        w_u_list = tf.split(w_u_, 1, axis=axis)
        w_l_list = tf.split(w_l_, 1, axis=axis)

    if mode == F_IBP.name:
        outputs = [[u_list[i], l_list[i]] for i in range(n)]
    elif mode == F_FORWARD.name:
        outputs = [[x_, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
    elif mode == F_HYBRID.name:
        outputs = [[x_, u_list[i], w_u_list[i], b_u_list[i], l_list[i], w_l_list[i], b_l_list[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return outputs


def sort(input_, axis=-1, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of sort by selection

    Args:
        input_
        axis
        dc_decomp
        convex_domain
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    # remove grad bounds

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_IBP.name:
        u_c, l_c = input_[:nb_tensor]
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = input_[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = input_[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if axis == -1:
        n = input_.shape[-1]
        axis = len(input_.shape) - 1
    else:
        n = input_.shape[axis]

    # what about splitting elements
    op = lambda x: tf.split(x, n, axis=axis)
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_list = op(u_c)
        l_c_list = op(l_c)

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        w_u_list = tf.split(w_u, n, axis=axis + 1)
        b_u_list = op(b_u)
        w_l_list = tf.split(w_l, n, axis=axis + 1)
        b_l_list = op(b_l)

    def get_input(mode, i):
        if mode == F_IBP.name:
            return [u_c_list[i], l_c_list[i]]
        if mode == F_FORWARD.name:
            return [x_0, w_u_list[i], b_u_list[i], w_l_list[i], b_l_list[i]]
        if mode == F_HYBRID.name:
            return [x_0, u_c_list[i], w_u_list[i], b_u_list[i], l_c_list[i], w_l_list[i], b_l_list[i]]

    def set_input(input_, mode, i):
        if mode == F_IBP.name:
            u_i, l_i = input_
        elif mode == F_FORWARD.name:
            _, w_u_i, b_u_i, w_l_i, b_l_i = input_
        elif mode == F_HYBRID.name:
            _, u_i, w_u_i, b_u_i, l_i, w_l_i, b_l_i = input_
        else:
            raise ValueError(f"Unknown mode {mode}")

        if mode in [F_IBP.name, F_HYBRID.name]:
            u_c_list[i] = u_i
            l_c_list[i] = l_i
        if mode in [F_FORWARD.name, F_HYBRID.name]:
            w_u_list[i] = w_u_i
            w_l_list[i] = w_l_i
            b_u_list[i] = b_u_i
            b_l_list[i] = b_l_i

    # use selection sort
    for i in range(n - 1):
        for j in range(i + 1, n):

            input_i = get_input(mode, i)
            input_j = get_input(mode, j)
            output_a = maximum(input_i, input_j, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)
            output_b = minimum(input_i, input_j, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)

            set_input(output_a, mode, j)
            set_input(output_b, mode, i)

    op_ = lambda x: K.concatenate(x, axis)
    # update the inputs
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = op_(u_c_list)
        l_c_ = op_(l_c_list)
    if mode in [F_FORWARD.name, F_HYBRID.name]:
        w_u_ = K.concatenate(w_u_list, axis + 1)
        w_l_ = K.concatenate(w_l_list, axis + 1)
        b_u_ = op_(b_u_list)
        b_l_ = op_(b_l_list)

    if mode == F_IBP.name:
        output = [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output = [x_0, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")
    return output


def pow(inputs_, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of pow(x )=x**2

    Args:
        inputs_
        dc_decomp
        convex_domain
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    return multiply(inputs_, inputs_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def abs(inputs_, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of |x|

    Args:
        inputs_
        dc_decomp
        convex_domain
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    inputs_0 = relu_(inputs_, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)
    inputs_1 = minus(
        relu_(minus(inputs_, mode=mode), dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode), mode=mode
    )

    return add(inputs_0, inputs_1, dc_decomp=dc_decomp, convex_domain=convex_domain, mode=mode)


def frac_pos_hull(inputs_, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name):
    """LiRPA implementation of 1/x for x>0

    Args:
        inputs_
        dc_decomp
        convex_domain
        mode

    Returns:

    """

    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()
    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors

    if mode == F_IBP.name:
        u_c, l_c = inputs_[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = inputs_[:nb_tensor]
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")
    if mode in [F_FORWARD.name, F_HYBRID.name]:
        u_c_ = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
        l_c_ = get_lower(x_0, w_u, b_u, convex_domain=convex_domain)

        if mode == F_FORWARD.name:
            u_c = u_c_
            l_c = l_c_
        else:
            u_c = K.minimum(u_c_, u_c)
            l_c = K.maximum(l_c, l_c_)

    l_c = K.maximum(l_c, 1.0)
    z = (u_c + l_c) / 2.0
    w_l = -1 / K.pow(z)
    b_l = 2 / z
    w_u = (1.0 / u_c - 1.0 / l_c) / (u_c - l_c)
    b_u = 1.0 / u_c - w_u * u_c

    return [w_u, b_u, w_l, b_l]


# convex hull for min
def min_(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, axis=-1, finetune=False, **kwargs):
    """LiRPA implementation of min(x, axis=axis)

    Args:
        x
        dc_decomp
        convex_domain
        mode
        axis

    Returns:

    """
    # return - max - x
    if convex_domain is None:
        convex_domain = {}
    return minus(
        max_(
            minus(x, mode=mode),
            dc_decomp=dc_decomp,
            convex_domain=convex_domain,
            mode=mode,
            axis=axis,
            finetune=finetune,
            **kwargs,
        ),
        mode=mode,
    )


def expand_dims(inputs_, dc_decomp=False, mode=F_HYBRID.name, axis=-1, **kwargs):

    nb_tensor = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_IBP.name:
        u_c, l_c = inputs_[:nb_tensor]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = inputs_[:nb_tensor]
    elif mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs_[:nb_tensor]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [F_HYBRID.name, F_FORWARD.name]:
        if axis == -1:
            axis_w = axis
        else:
            axis_w = axis + 1

    op = lambda t: K.expand_dims(t, axis)

    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = op(u_c)
        l_c_ = op(l_c)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        b_u_ = op(b_u)
        b_l_ = op(b_l)
        w_u_ = K.expand_dims(w_u, axis_w)
        w_l_ = K.expand_dims(w_l, axis_w)

    if mode == F_IBP.name:
        output = [u_c_, l_c_]
    elif mode == F_FORWARD.name:
        output = [x_0, w_u_, b_u_, w_l_, b_l_]
    elif mode == F_HYBRID.name:
        output = [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")

    return output


def log(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """Exponential activation function.

    Args:
        x: list of input tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex input domain
        mode: type of Forward propagation (IBP, Forward or Hybrid)
        **kwargs: extra parameters
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        u_c, l_c = x
        return [K.log(u_c), K.log(l_c)]
    elif mode in [F_FORWARD.name, F_HYBRID.name]:
        if mode == F_FORWARD.name:
            x_0, w_u, b_u, w_l, b_l = x
            u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
            l_c = K.maximum(K.epsilon(), l_c)
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

        u_c_ = K.log(u_c)
        l_c_ = K.log(l_c)

        y = (u_c + l_c) / 2.0

        w_l_0 = (u_c_ - l_c_) / K.maximum(u_c - l_c, K.epsilon())
        b_l_0 = l_c_ - w_l_0 * l_c

        w_u_0 = 1 / y
        b_u_0 = K.log(y) - 1

        w_u_ = w_u_0[:, None] * w_u
        b_u_ = w_u_0 * b_u + b_u_0
        w_l_ = w_l_0[:, None] * w_l
        b_l_ = w_l_0 * b_l + b_l_0

        if mode == F_FORWARD.name:
            return [x_0, w_u_, b_u_, w_l_, b_l_]
        else:
            return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")


def exp(x, dc_decomp=False, convex_domain=None, mode=F_HYBRID.name, **kwargs):
    """Exponential activation function.

    Args:
        x: list of input tensors
        dc_decomp: boolean that indicates
        convex_domain: the type of convex input domain
        mode: type of Forward propagation (IBP, Forward or Hybrid)
        **kwargs: extra parameters
    whether we return a difference of convex decomposition of our layer

    Returns:
        the updated list of tensors
    """
    if convex_domain is None:
        convex_domain = {}
    if dc_decomp:
        raise NotImplementedError()

    if mode == F_IBP.name:
        return [K.exp(e) for e in x]
    elif mode in [F_FORWARD.name, F_HYBRID.name]:
        if mode == F_FORWARD.name:
            x_0, w_u, b_u, w_l, b_l = x
            u_c = get_upper(x_0, w_u, b_u, convex_domain=convex_domain)
            l_c = get_lower(x_0, w_l, b_l, convex_domain=convex_domain)
        else:
            x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

        u_c_ = K.exp(u_c)
        l_c_ = K.exp(l_c)

        y = (u_c + l_c) / 2.0  # do finetuneting

        w_u_0 = (u_c_ - l_c_) / K.maximum(u_c - l_c, K.epsilon())
        b_u_0 = l_c_ - w_u_0 * l_c

        w_l_0 = K.exp(y)
        b_l_0 = w_l_0 * (1 - y)

        w_u_ = w_u_0[:, None] * w_u
        b_u_ = w_u_0 * b_u + b_u_0
        w_l_ = w_l_0[:, None] * w_l
        b_l_ = w_l_0 * b_l + b_l_0

        if mode == F_FORWARD.name:
            return [x_0, w_u_, b_u_, w_l_, b_l_]
        else:
            return [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]
    else:
        raise ValueError(f"Unknown mode {mode}")
