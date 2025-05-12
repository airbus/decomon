from typing import Any

import keras.ops as K
from jacobinet.layers.convolutional.conv1d import BackwardConv1D

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer
from decomon.layers.convolutional.utils import ConvKernelConstraint
from decomon.layers.utils import pre_built


class DecomonBackwardConv1D(DecomonBackwardLinearLayer):
    layer: BackwardConv1D

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward_pos = ConvKernelConstraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = ConvKernelConstraint(layer=self.layer_backward, ops=K.minimum, add_bias=False)

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardConv1D(self.layer_backward_pos)
        self.layer_neg = BackwardConv1D(self.layer_backward_neg)
