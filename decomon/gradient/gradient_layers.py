from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Permute, Activation
from decomon.layers.decomon_layers import (
    DecomonDense,
    DecomonConv2D,
    DecomonDropout,
    DecomonReshape,
    DecomonFlatten,
    DecomonBatchNormalization,
    DecomonActivation,
    DecomonPermute,
)
from ..backward_layers.activations import get
from tensorflow.keras.backend import conv2d, conv2d_transpose
from .utils import V_slope, backward_sort, get_identity_lirpa, get_IBP, get_FORWARD, get_input_dim
from ..layers.utils import ClipAlpha, F_HYBRID, F_FORWARD
from ..layers.decomon_layers import to_monotonic
from .backward_maxpooling import BackwardMaxPooling2D
from .backward_merge import BackwardAverage
from tensorflow.python.ops import array_ops

class GradientDense(Layer):
    """
    Gradient LiRPA of Dense Layer
    """

    def __init__(
        self,
        layer,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain={},
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super(GradientDense, self).__init__(**kwargs)

        
        self.layer = layer

    def call(self, inputs, mode, convex_domain):

        return self.kernel