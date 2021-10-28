from __future__ import absolute_import
import tensorflow as tf
from .core import DecomonLayer
import tensorflow.keras.backend as K
from tensorflow.keras.backend import bias_add, conv2d
import numpy as np
from tensorflow.keras.constraints import NonNeg

# from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Activation,
    Flatten,
    Reshape,
    Dot,
    Input,
    BatchNormalization,
    Dropout,
    Lambda,
    InputSpec,
)
from tensorflow.python.keras.layers.merge import _Merge as Merge
from decomon.layers import activations
from decomon.layers.utils import NonPos, MultipleConstraint, Project_initializer_pos, Project_initializer_neg
from tensorflow.python.keras.utils import conv_utils
from decomon.layers.utils import get_upper, get_lower, sort
from .maxpooling import DecomonMaxPooling2D
from .decomon_merge_layers import (
    DecomonConcatenate,
    DecomonAverage,
    DecomonAdd,
    DecomonMinimum,
    DecomonMaximum,
    DecomonSubtract,
    to_monotonic_merge,
)
from .utils import grad_descent
from .core import F_FORWARD, F_IBP, F_HYBRID

from deel.lip.layers import (
    ScaledL2NormPooling2D,
    FrobeniusDense,
)

from .decomon_reshape import DecomonReshape
from .decomon_merge_layers import DecomonConcatenate
from .utils import sort, max_, min_, expand_dims


from deel.lip.model import Sequential
from deel.lip.activations import GroupSort


class DecomonGroupSort(DecomonLayer):
    def __init__(self, n=None, data_format="channels_last", k_coef_lip=1.0, mode=F_HYBRID.name, **kwargs):
        super(DecomonGroupSort, self).__init__(**kwargs)
        self.mode = mode

        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError("channels_first not implemented for GroupSort activation")
            self.channel_axis = 1
        else:
            raise RuntimeError("data format not understood")
        self.n = n
        self.reshape = DecomonReshape(
            (-1, self.n), mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call
        self.concat = DecomonConcatenate(
            mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call

    def call(self, input):

        shape_in = list(input[0].shape[1:])

        if self.mode == F_IBP.name:
            y, x_0, u_c, l_c = input[:4]
        if self.mode == F_FORWARD.name:
            y, x_0, w_u, b_u, w_l, b_l = input[:6]
        if self.mode == F_HYBRID.name:
            y, x_0, u_c, w_u, b_u, l_c, w_l, b_l = input[:8]

        input_ = self.reshape(input)
        import pdb

        pdb.set_trace()

        if self.n == 2:

            output_max = expand_dims(
                max_(input_, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode, axis=-1),
                dc_decomp=self.dc_decomp,
                mode=self.mode,
                axis=-1,
            )
            output_min = expand_dims(
                min_(input_, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode, axis=-1),
                dc_decomp=self.dc_decomp,
                mode=self.mode,
                axis=-1,
            )
            output_ = self.concat([output_min, output_max])

        else:

            output_ = sort(input_, axis=-1, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode)

        return DecomonReshape(
            shape_in, mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call(output_)

    def compute_output_shape(self, input_shape):
        return input_shape
