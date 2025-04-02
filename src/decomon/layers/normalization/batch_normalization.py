from keras.layers import BatchNormalization #type:ignore

from decomon.layers.layer import DecomonLinearLayer
from decomon.layers.normalization.utils import BatchNormalization_kernel_constraint
import keras.ops as K #type:ignore

class DecomonBatchNormalization(DecomonLinearLayer):
    layer: BatchNormalization
    linear = True
    diagonal = True
    use_bias = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_pos = BatchNormalization_kernel_constraint(layer=self.layer, ops=K.maximum, center=self.layer.center)
        self.layer_neg = BatchNormalization_kernel_constraint(layer=self.layer, ops=K.minimum, center=False)