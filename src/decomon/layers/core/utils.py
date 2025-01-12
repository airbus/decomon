import warnings

import keras.ops as K
from keras.layers import Dense
from keras.layers import Wrapper

from typing import Optional, Any
from decomon.types import Tensor

class Dense_kernel_constraint(Wrapper):
    def __init__(self, layer:Dense, ops=K.maximum, add_bias=True, **kwargs:Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias
    
    def call(self, inputs: list[Tensor]) -> list[Tensor]:
        y:Tensor =  K.matmul(inputs, self.ops(0, self.layer.kernel))

        if self.layer.bias is not None and self.add_bias:
            y = K.add(y, self.layer.bias)

        return y