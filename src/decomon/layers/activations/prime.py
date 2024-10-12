import keras.ops as K
from decomon.types import Tensor


def leaky_relu_prime(x: Tensor, negative_slope: float) -> Tensor:
    # Derivative of leaky relu: 1 if x_i>=0 and negative_slope if x_i <= 0
    # 1 if x>=0
    # negative_slope if x<=0
    mask = K.relu(K.sign(x))
    return mask + (1 - mask) * negative_slope
