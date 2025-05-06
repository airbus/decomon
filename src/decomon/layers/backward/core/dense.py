from typing import Any

import keras.ops as K
from jacobinet.layers.core.dense import BackwardDense

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward
from decomon.layers.core.utils import DenseKernelConstraint
from decomon.layers.utils import pre_built


class DecomonBackwardDense(DecomonLinearLayerBackward):
    layer: BackwardDense

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward_pos = DenseKernelConstraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = DenseKernelConstraint(layer=self.layer_backward, ops=K.minimum, add_bias=False)

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardDense(self.layer_backward_pos)
        self.layer_neg = BackwardDense(self.layer_backward_neg)
