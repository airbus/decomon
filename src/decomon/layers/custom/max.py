# define non native class Max
# Decomon Custom for Max(axis...)
import keras
from decomon.layers.custom.utils import get_affine_lower_bound_max, get_affine_upper_bound_max
from decomon.layers import DecomonLayer
from decomon.types import Tensor
import keras.ops as K
import numpy as np

from typing import Tuple, List


class Max(keras.layers.Layer):
    """Custom Keras Layer that computes max on a Keras Tensor.
    """

    def __init__(self, axis:int, keepdims=True, **kwargs):
        """
        Compute the result of  max(x) along a given axis: K.max(x, axis)
        Args:
            axis: The axis dimension on which we perform the operator.
        """
        super(Max, self).__init__(**kwargs)
        self.axis = axis
        self.keepdims=keepdims

    def call(self, inputs_):
        return keras.ops.max(inputs_, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis":  keras.saving.serialize_keras_object(self.axis),
                "keepdims": keras.saving.serialize_keras_object(self.keepdims),
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        
        input_shape = list(input_shape)
        if self.axis<0:
            axis_ = len(input_shape)+self.axis
        else:
            axis_ = self.axis

        if self.keepdims:
            tmp_shape=[1]
        else:
            tmp_shape=[]
        return input_shape[:axis_]+tmp_shape+input_shape[axis_+1:]

    @classmethod
    def from_config(cls, config):
        axis_config = config.pop('axis')
        keepdims_config = config.pop('keepdims')
        axis = keras.saving.deserialize_keras_object(axis_config)
        keepdims = keras.saving.deserialize_keras_object(keepdims_config)
        return cls(axis=axis, keepdims=keepdims, **config)


class DecomonMax(DecomonLayer):

    layer: Max
    linear=False

    def get_affine_bounds(self, lower:Tensor, upper:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        w_l:Tensor
        b_l:Tensor
        w_u:Tensor
        b_u:Tensor
        diag_:Tensor

        # compute positive value for axis
        axis_:int
        if self.layer.axis<0:
            axis_ = len(lower.shape)+self.layer.axis
        else:
            axis_ = self.layer.axis

        # compute affine bounds for max(x, axis)
        input_shape:List[int] = list(self.layer.input.shape)
        input_shape_wo_batch:List[int] = input_shape[1:]
        output_shape_wo_batch:List[int] = list(self.layer.output.shape[1:])
        
        w_l, b_l = get_affine_lower_bound_max(lower, upper, axis=self.layer.axis, keepdims=self.layer.keepdims)

        w_u, b_u = get_affine_upper_bound_max(lower, upper, axis=self.layer.axis, keepdims=self.layer.keepdims)

        input_shape_wo_axis:int = np.prod(output_shape_wo_batch)
        
        if self.layer.keepdims:
            diag_ = K.reshape(K.eye(input_shape_wo_axis), [1]+output_shape_wo_batch+output_shape_wo_batch)
        else:
            #output_shape_broadcast_axis = output_shape_wo_batch[:axis_-1]+[1]+output_shape_wo_batch[axis_:]
            output_shape_broadcast_axis:List[int] = input_shape_wo_batch[:axis_-1]+input_shape_wo_batch[axis_:]
            target_shape:List[int] = [1]+output_shape_broadcast_axis+output_shape_wo_batch
            diag_ = K.reshape(K.eye(input_shape_wo_axis), target_shape)
            
        if not self.layer.keepdims:
            diag_ = K.expand_dims(diag_, axis_)

        N_out:int = len(output_shape_wo_batch)
            
        w_l = K.reshape(w_l, [-1]+input_shape_wo_batch+[1]*N_out)
        w_l = diag_*w_l
        w_u = K.reshape(w_u, [-1]+input_shape_wo_batch+[1]*N_out)
        w_u = diag_*w_u
        
        return w_l, b_l, w_u, b_u

