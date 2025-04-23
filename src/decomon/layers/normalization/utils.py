from typing import Any

import keras.ops as K  # type:ignore
from keras.layers import BatchNormalization  # type:ignore
from keras.layers import Wrapper  # type:ignore
from keras.src import backend  # type:ignore
from keras.src import ops  # type:ignore


class BatchNormalization_kernel_constraint(Wrapper):
    def __init__(self, layer: BatchNormalization, ops=K.maximum, add_moving_mean=True, center=True, **kwargs: Any):
        super().__init__(layer=layer, **kwargs)
        self.ops = ops
        self.add_moving_mean = add_moving_mean
        self.moving_mean_ = self.layer.moving_mean
        self.moving_variance = self.layer.moving_variance
        self.scale = self.layer.scale
        self.gamma_ = self.layer.gamma
        self.center = center
        self.beta = self.layer.beta
        self.axis = self.layer.axis
        self.epsilon = self.layer.epsilon

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    @property
    def gamma(self):
        return self.ops(self.gamma_, 0)

    @property
    def moving_mean(self):
        if self.add_moving_mean:
            return self.moving_mean_
        else:
            return 0.0 * self.moving_mean_

    def call(self, inputs, training=None, mask=None):
        # Check if the mask has one less dimension than the inputs.
        if mask is not None:
            if len(mask.shape) != len(inputs.shape) - 1:
                # Raise a value error
                raise ValueError(
                    "The mask provided should be one dimension less "
                    "than the inputs. Received: "
                    f"mask.shape={mask.shape}, inputs.shape={inputs.shape}"
                )

        compute_dtype = backend.result_type(inputs.dtype, "float32")
        # BN is prone to overflow with float16/bfloat16 inputs, so we upcast to
        # float32 for the subsequent computations.
        inputs = ops.cast(inputs, compute_dtype)

        moving_mean = ops.cast(self.moving_mean, inputs.dtype)
        moving_variance = ops.cast(self.moving_variance, inputs.dtype)

        mean = moving_mean
        variance = moving_variance

        if self.scale:
            gamma = ops.cast(self.gamma, inputs.dtype)
            # apply ops
        else:
            gamma = None

        if self.center:
            beta = ops.cast(self.beta, inputs.dtype)
        else:
            beta = None

        outputs = ops.batch_normalization(
            x=inputs,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        return ops.cast(outputs, self.layer.compute_dtype)
