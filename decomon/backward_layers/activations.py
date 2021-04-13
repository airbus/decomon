from __future__ import absolute_import
import warnings
import six
from ..backward_layers.utils import backward_relu_
from .utils import V_slope
from tensorflow.keras.layers import Layer


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
    grad_bounds=False,
    convex_domain={},
    alpha=0.0,
    max_value=None,
    threshold=0.0,
    slope=V_slope.name,
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
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    if threshold != 0:
        raise NotImplementedError()

    if not (alpha) and max_value is None:
        # default values: return relu_(x) = max(x, 0)
        y = x[:-2]
        w_out_u, b_out_u, w_out_l, b_out_l = x[-4:]

        return backward_relu_(y, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain=convex_domain)
    raise NotImplementedError()


def backward_sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def backward_tanh(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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
    if dc_decomp or grad_bounds:
        raise NotImplementedError()

    # TO DO linear relaxation
    raise NotImplementedError()


def bacward_hard_sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_elu(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_selu(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_linear(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_exponential(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_softplus(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
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


def backward_softsign(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name):
    """Softsign activation function `x / (abs(x) + 1)`.

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


def backward_softmax(x, dc_decomp=False, grad_bounds=False, convex_domain={}, slope=V_slope.name, axis=-1):
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
        return backward_relu_
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
