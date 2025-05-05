import keras.ops as K
from jacobinet.layers.convolutional.depthwise_conv2d import BackwardDepthwiseConv2D

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward
from decomon.layers.convolutional.utils import DepthwiseConv_kernel_constraint
from decomon.layers.utils import pre_built


class DecomonBackwardDepthwiseConv2D(DecomonLinearLayerBackward):
    layer: BackwardDepthwiseConv2D

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward_pos = DepthwiseConv_kernel_constraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = DepthwiseConv_kernel_constraint(
            layer=self.layer_backward, ops=K.minimum, add_bias=False
        )

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardDepthwiseConv2D(self.layer_backward_pos)
        self.layer_neg = BackwardDepthwiseConv2D(self.layer_backward_neg)
