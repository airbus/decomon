from typing import Any

import keras

from decomon.layers import DecomonLinearLayer
from decomon.layers.merging.base_merge import DecomonMerge
from decomon.layers.utils import pre_built


class DecomonFuseGradients(DecomonMerge):
    linear = True
    increasing = True


class DecomonGradConstant(DecomonLinearLayer):
    linear = True
    increasing = True

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward = keras.layers.Lambda(lambda x: 0 * x)
        # pre built the layers
        pre_built(self.layer_backward, self.layer.input_dim_wo_batch)
