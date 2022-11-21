import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.backend import conv2d, conv2d_transpose
from tensorflow.keras.layers import Activation, Flatten, Layer, Permute
from tensorflow.python.ops import array_ops

from decomon.backward_layers.activations import get
from decomon.layers.core import F_FORWARD, F_HYBRID
from decomon.layers.decomon_layers import (
    DecomonActivation,
    DecomonBatchNormalization,
    DecomonConv2D,
    DecomonDense,
    DecomonDropout,
    DecomonFlatten,
    DecomonPermute,
    DecomonReshape,
    to_monotonic,
)


class GradientDense(Layer):
    """
    Gradient LiRPA of Dense Layer
    """

    def __init__(
        self,
        layer,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain=None,
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if convex_domain is None:
            convex_domain = {}
        self.layer = layer

    def call(self, *args, **kwargs):

        return self.kernel
