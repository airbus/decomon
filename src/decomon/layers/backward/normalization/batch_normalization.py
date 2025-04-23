import keras
import keras.ops as K
from jacobinet.layers.normalization.batch_normalization import (
    BackwardBatchNormalization,  # type: ignore
)
from keras.src import ops

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward
from decomon.layers.normalization.utils import BatchNormalization_kernel_constraint
from decomon.layers.utils import pre_built


class DecomonBackwardBatchNormalization(DecomonLinearLayerBackward):
    layer: BackwardBatchNormalization
    diagonal = True
    use_bias = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create positive and negative version
        self.layer_backward_pos = BatchNormalization_kernel_constraint(layer=self.layer_backward, ops=K.maximum)
        self.layer_backward_neg = BatchNormalization_kernel_constraint(
            layer=self.layer_backward, ops=K.minimum, add_bias=False
        )

        # pre built the layers

        pre_built(self.layer_backward_pos, self.layer.input_dim_wo_batch)
        pre_built(self.layer_backward_neg, self.layer.input_dim_wo_batch)

        self.layer_pos = BackwardBatchNormalization(self.layer_backward_pos)
        self.layer_neg = BackwardBatchNormalization(self.layer_backward_neg)
