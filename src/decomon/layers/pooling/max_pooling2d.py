from decomon.layers import DecomonLayer
from decomon.types import Tensor

import keras
from keras.layers import Layer, Reshape, MaxPooling2D
from keras.models import Sequential
import keras.ops as K
import numpy as np

from decomon.layers.custom.utils import get_affine_lower_bound_max, get_affine_upper_bound_max
from decomon.layers.convolutional.utils import get_toeplitz

from typing import Optional, Any

from decomon.constants import Propagation
from decomon.perturbation_domain import PerturbationDomain

from .utils_conv import get_conv_op_config, get_conv_op, get_in_channels


class DecomonMaxPooling2D(DecomonLayer):

    layer: MaxPooling2D
    linear: False

    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            **kwargs,
        )

        # build kernel and depthwise
        in_channels = get_in_channels(self.layer)
        conv_op, kernel = get_conv_op(self.layer)
        self.conv_op = conv_op
        self.kernel = kernel[:, :, :1]

        # build reshape layer
        input_shape_wo_batch = list(self.layer.input.shape[1:])

        input_shape_toeplitz = [e for e in input_shape_wo_batch]
        output_shape_toeplitz = [e for e in input_shape_wo_batch]

        if self.layer.data_format == "channels_first":
            target_shape = [input_shape_wo_batch[0], -1] + input_shape_wo_batch[1:]
            input_shape_toeplitz[0] = 1
            output_shape_toeplitz[0] = self.conv_op.depth_multiplier
            self.axis = 2
        else:
            target_shape = input_shape_wo_batch + [-1]
            target_shape[-1] = -1
            input_shape_toeplitz[-1] = 1
            output_shape_toeplitz[-1] = self.conv_op.depth_multiplier
            self.axis = -1

        self.reshape_op = Reshape(target_shape)

        self.inner_model = Sequential([self.conv_op, self.reshape_op])
        _ = self.inner_model(self.layer.input)

        config = self.conv_op.get_config()
        config["kernel_size"] = self.layer.pool_size

        self.matrix = get_toeplitz(self.kernel, input_shape_toeplitz, output_shape_toeplitz, config)
        var = K.eye(in_channels)

        if self.layer.data_format == "channels_first":
            var = K.reshape(
                var,
                [in_channels]
                + [1] * (len(input_shape_toeplitz) - 1)
                + [in_channels]
                + [1] * len(output_shape_toeplitz),
            )
            self.matrix = K.reshape(self.matrix, input_shape_toeplitz + [1] + output_shape_toeplitz)
        else:
            raise NotImplementedError()

        self.matrix = K.expand_dims(self.matrix * var, 0)

    def get_affine_bounds(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        upper_max = self.reshape_op(self.conv_op(upper))
        lower_max = self.reshape_op(self.conv_op(lower))
        bias = 0.0 * self.layer(lower)

        # compute affine bounds for max(x, axis)
        w_l_max, _ = get_affine_lower_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False)
        w_u_max, b_u = get_affine_upper_bound_max(lower_max, upper_max, axis=self.axis, keepdims=False)

        w_l_max = w_l_max[:, None, None, None]

        w_u_max = w_u_max[:, None, None, None]

        w_u = K.sum(self.matrix * w_u_max, axis=len(self.layer.input.shape) - 1 + self.axis)

        w_l = K.sum(self.matrix * w_l_max, axis=len(self.layer.input.shape) - 1 + self.axis)
        b_l = bias

        return w_l, b_l, w_u, b_u
