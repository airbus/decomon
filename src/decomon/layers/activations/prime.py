import keras.ops as K
from decomon.types import Tensor


def softplus_prime(x:Tensor)->Tensor:

    #1/ (1 +exp(−x ))

    return 1/(1 + K.exponential(-x))

def elu_prime(x:Tensor, alpha:float=1.0)->Tensor:
    # Derivative of elu: x if x>=0 and alpha*(exp(x)-1) if x<0
    # 1 if x>=0
    # alpha*exp(x) = elu(x)+1 if x<=0
    mask = K.relu(K.sign(x))
    return mask + (1 - mask) * (K.elu(x)+1)


def selu_prime(x:Tensor, alpha:float=1.0)->Tensor:
    # selu = scale*elu(x, alpha)
    # alpha=1.67326324` and `scale=1.05070098
    alpha = 1.67326324
    scale = 1.05070098
    mask = K.relu(K.sign(x))
    return scale*(mask + (1 - mask) * (elu(x, alpha=alpha)+1))
    
def leaky_relu_prime(x: Tensor, negative_slope: float) -> Tensor:
    # Derivative of leaky relu: 1 if x_i>=0 and negative_slope if x_i <= 0
    # 1 if x>=0
    # negative_slope if x<=0
    mask = K.relu(K.sign(x))
    return mask + (1 - mask) * negative_slope


def sigmoid_prime(x: Tensor) -> Tensor:
    """Derivative of sigmoid

    Args:
        x

    Returns:

    """

    s_x = K.sigmoid(x)
    return s_x * (K.cast(1, dtype=x.dtype) - s_x)


def tanh_prime(x: Tensor) -> Tensor:
    """Derivative of tanh

    Args:
        x

    Returns:

    """

    s_x = K.tanh(x)
    return K.cast(1, dtype=x.dtype) - K.power(s_x, K.cast(2, dtype=x.dtype))


def relu_prime(x: Tensor) -> Tensor:
    """Derivative of relu

    Args:
        x

    Returns:

    """

    return K.clip(K.sign(x), K.cast(0, dtype=x.dtype), K.cast(1, dtype=x.dtype))


def softsign_prime(x: Tensor) -> Tensor:
    """Derivative of softsign

    Args:
        x

    Returns:

    """

    return K.cast(1.0, dtype=x.dtype) / K.power(K.cast(1.0, dtype=x.dtype) + K.abs(x), K.cast(2, dtype=x.dtype))


