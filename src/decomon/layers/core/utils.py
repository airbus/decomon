from collections.abc import Callable
from typing import Any

import keras
import keras.ops as K
from keras.layers import Dense, Wrapper

from decomon.types import Tensor


class Dense_kernel_constraint(Wrapper):
    def __init__(
        self, layer: Dense, ops: Callable[[Tensor, Tensor], Tensor] = K.maximum, add_bias: bool = True, **kwargs: Any
    ):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias

    @property
    def kernel(self) -> keras.Variable:
        return self.ops(0, self.layer.kernel)

    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        y: Tensor = K.matmul(inputs, self.ops(0, self.layer.kernel))

        if self.layer.bias is not None and self.add_bias:
            y = K.add(y, self.layer.bias)

        return y
