from typing import Any

import keras.ops as K
from jacobinet.layers.convolutional.conv3d import BackwardConv3D

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward
from decomon.layers.convolutional.utils import Conv_kernel_constraint
from decomon.layers.utils import pre_built


class DecomonBackwardConv3D(DecomonLinearLayerBackward):
    layer: BackwardConv3D

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward_pos = Conv_kernel_constraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = Conv_kernel_constraint(layer=self.layer_backward, ops=K.minimum, add_bias=False)

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardConv3D(self.layer_backward_pos)
        self.layer_neg = BackwardConv3D(self.layer_backward_neg)
