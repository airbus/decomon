import keras
import numpy as np
from jacobinet.models.utils import FuseGradients, GradConstant
from keras.models import Sequential

from decomon.layers import DecomonLinearLayer
from decomon.layers.merging.base_merge import DecomonMerge
from decomon.layers.utils import pre_built


class DecomonFuseGradients(DecomonMerge):
    linear = True
    increasing = True


def pre_built(layer, input_shape_wo_batch):
    if not layer.built:
        toy_model = Sequential([layer])
        input = np.zeros([1] + input_shape_wo_batch)
        _ = toy_model(input)


class DecomonGradConstant(DecomonLinearLayer):
    linear = True
    increasing = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # create positive and negative version

        self.layer_backward = keras.layers.Lambda(lambda x: 0 * x)
        # pre built the layers
        pre_built(self.layer_backward, self.layer.input_dim_wo_batch)
