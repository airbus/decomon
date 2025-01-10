from typing import Optional, Any

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Layer, Wrapper
import keras.ops as K
from decomon.constants import Propagation
from decomon.perturbation_domain import BoxDomain, PerturbationDomain
from decomon.layers.convolutional.utils import get_toeplitz_from_layer as get_toeplitz
from decomon.layers.utils import get_bias


from decomon.layers.layer import DecomonLayer, DecomonLinearLayer
from typing import Optional
from decomon.types import Tensor

import numpy as np


class Conv_kernel_constraint(Wrapper):

    def __init__(self, layer:Layer, ops=K.maximum, add_bias=True, **kwargs:Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_bias = add_bias
    
    def call(self, inputs: list[Tensor]) -> list[Tensor]:

        y:Tensor =  K.conv(
            inputs,
            kernel = self.ops(0, self.layer.kernel),
            strides=list(self.layer.strides),
            padding=self.layer.padding,
            dilation_rate=self.layer.dilation_rate,
            data_format=self.layer.data_format,
        ) 

        if self.add_bias:
            y += self.layer(0*inputs)

        return y

class DecomonConv2D(DecomonLinearLayer):

    # create layer_increasing, layer_decreasing
    def __init__(
        self,
        layer:Conv2D,
        *args,
        **kwargs: Any,
    ):
        layer_pos = Conv_kernel_constraint(layer=layer, ops=K.maximum, add_bias=True)
        layer_neg = Conv_kernel_constraint(layer=layer, ops=K.minimum, add_bias=False)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args,**kwargs)
        