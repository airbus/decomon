from typing import Any

import keras.ops as K
from keras.layers import BatchNormalization

from decomon.layers.layer import DecomonLayer
from decomon.layers.normalization.utils import BatchNormalizationKernelConstraint


class DecomonBatchNormalization(DecomonLayer):
    layer: BatchNormalization
    linear = True
    diagonal = True
    use_bias = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_pos = BatchNormalizationKernelConstraint(layer=self.layer, ops=K.maximum)
        self.layer_neg = BatchNormalizationKernelConstraint(layer=self.layer, ops=K.minimum, null_if_noscale=True)
