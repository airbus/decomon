from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from deel.lip.layers import FrobeniusDense, ScaledL2NormPooling2D
from tensorflow.keras.backend import bias_add, conv2d
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dot,
    Dropout,
    Flatten,
    Input,
    InputSpec,
    Lambda,
    Reshape,
)
from tensorflow.python.keras.layers.merge import _Merge as Merge
from tensorflow.python.keras.utils import conv_utils

from decomon.layers import activations
from decomon.layers.utils import (
    MultipleConstraint,
    NonPos,
    Project_initializer_neg,
    Project_initializer_pos,
    get_lower,
    get_upper,
    sort,
)

from .core import F_FORWARD, F_HYBRID, F_IBP, DecomonLayer
from .decomon_merge_layers import (
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonMaximum,
    DecomonMinimum,
    DecomonSubtract,
    to_monotonic_merge,
)
from .decomon_reshape import DecomonReshape
from .maxpooling import DecomonMaxPooling2D
from .utils import ClipAlpha, expand_dims, grad_descent, max_, min_, sort

try:
    from deel.lip.activations import GroupSort, GroupSort2
except:
    raise Warning(
        "Could not import GroupSort from deel.lip.activations. Install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


class DecomonGroupSort(DecomonLayer):
    def __init__(self, n=None, data_format="channels_last", k_coef_lip=1.0, mode=F_HYBRID.name, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

        if data_format == "channels_last":
            self.channel_axis = -1
        elif data_format == "channels_first":
            raise RuntimeError("channels_first not implemented for GroupSort activation")
        else:
            raise RuntimeError("data format not understood")
        self.n = n
        self.reshape = DecomonReshape(
            (-1, self.n), mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call
        self.concat = DecomonConcatenate(
            mode=self.mode, convex_domain=self.convex_domain, dc_decomp=self.dc_decomp
        ).call

    def call(self, input, **kwargs):

        shape_in = list(input[0].shape[1:])
        input_ = self.reshape(input)
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


class DecomonGroupSort2(DecomonLayer):
    def __init__(self, n=2, data_format="channels_last", k_coef_lip=1.0, mode=F_HYBRID.name, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.data_format = data_format

        if self.data_format == "channels_last":
            self.axis = -1
        else:
            self.axis = 1

        if self.dc_decomp:
            raise NotImplementedError()

        self.op_concat = DecomonConcatenate(self.axis, mode=self.mode, convex_domain=self.convex_domain)
        self.op_reshape_in = None
        self.op_reshape_out = None

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):

        inputs_ = self.op_reshape_in(inputs)
        inputs_max = expand_dims(
            max_(
                inputs_,
                mode=self.mode,
                convex_domain=self.convex_domain,
                axis=self.axis,
                finetune=self.finetune,
                finetune_params=self.params_max,
            ),
            mode=self.mode,
            axis=self.axis,
        )
        inputs_min = expand_dims(
            min_(
                inputs_,
                mode=self.mode,
                convex_domain=self.convex_domain,
                axis=self.axis,
                finetune=self.finetune,
                finetune_params=self.params_min,
            ),
            mode=self.mode,
            axis=self.axis,
        )
        output = self.op_concat(inputs_min + inputs_max)
        output_ = self.op_reshape_out(output)
        return output_

    def build(self, input_shape):
        input_shape = input_shape[-1]

        if self.data_format == "channels_last":
            if input_shape[-1] % 2 != 0:
                raise ValueError()
            target_shape = input_shape[1:-2] + [int(input_shape[-1] / 2), 2]
        else:
            if input_shape[1] % 2 != 0:
                raise ValueError()
            target_shape = [2, int(input_shape[1] / 2)] + input_shape[2:]

        self.params_max = []
        self.params_min = []

        if self.finetune and self.mode in [F_FORWARD.name, F_HYBRID.name]:
            self.beta_max_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.beta_min_ = self.add_weight(
                shape=target_shape, initializer="ones", name="beta_max", regularizer=None, constraint=ClipAlpha()
            )
            self.params_max = [self.beta_max_]
            self.params_min = [self.beta_min_]

        self.op_reshape_in = DecomonReshape(target_shape, mode=self.mode)
        self.op_reshape_out = DecomonReshape(input_shape[1:], mode=self.mode)

    def reset_layer(self, layer):
        pass
