from typing import Any

import keras.ops as K
from jacobinet.layers.normalization.batch_normalization import (
    BackwardBatchNormalization,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer
from decomon.layers.normalization.utils import BatchNormalizationKernelConstraint
from decomon.layers.utils import pre_built


class DecomonBackwardBatchNormalization(DecomonBackwardLinearLayer):
    layer: BackwardBatchNormalization
    diagonal = True
    use_bias = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version
        self.layer_backward_pos = BatchNormalizationKernelConstraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = BatchNormalizationKernelConstraint(layer=self.layer_backward, ops=K.minimum)

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardBatchNormalization(self.layer_backward_pos)
        self.layer_neg = BackwardBatchNormalization(self.layer_backward_neg)
