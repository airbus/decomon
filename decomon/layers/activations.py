from __future__ import absolute_import
from .utils import *
import warnings
import six
from decomon.layers.core import DecomonLayer


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
        return relu_(x, dc_decomp=dc_decomp, grad_bounds=grad_bounds, convex_domain=convex_domain)

    raise NotImplementedError()


def sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def tanh(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def hard_sigmoid(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def elu(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def selu(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def linear(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def exponential(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def softplus(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def softsign(x, dc_decomp=False, grad_bounds=False, convex_domain={}):
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


def softmax(x, dc_decomp=False, grad_bounds=False, convex_domain={}, axis=-1):
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
