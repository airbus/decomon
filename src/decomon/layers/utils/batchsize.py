"""Adding batchsize to batch-independent outputs."""


from typing import Any, Optional

import keras
import keras.ops as K
from keras.layers import Layer

from decomon.types import BackendTensor


class InsertBatchAxis(Layer):
    """Layer adding a batch axis to its inputs.

    It accepts a list of tensors as its inputs. The convention is that:
    - The first tensor contains the batch axis (this we can deduce from it the batchsize)
    - The following tensors may miss or not this batch axis

    It returns the tensors, except the first ones,
    with a new first axis repeated to the proper batchsize when it was missing.

    """

    def call(self, inputs: list[BackendTensor]) -> list[BackendTensor]:
        batchsize = inputs[0].shape[0]

        outputs = [
            K.repeat(input_i[None], batchsize, axis=0) if missing_batchsize_i else input_i
            for input_i, missing_batchsize_i in zip(inputs[1:], self.missing_batchaxis[1:])
        ]

        return outputs

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        # called on list?
        if not isinstance(input_shape[0], (tuple, list)):
            raise ValueError(
                "An InsertBatchAxis layer should be called on a list of inputs. "
                f"Received: input_shape={input_shape} (not a list of shapes)"
            )
        # check first input has a batch axis
        if None not in input_shape[0]:
            raise ValueError("The first tensor in InsertBatchAxis inputs is supposed to have a batch axis.")
        # store the input indices missing the batchaxis
        self.missing_batchaxis = [None not in input_shape_i for input_shape_i in input_shape]
        self.built = True

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> list[tuple[Optional[int], ...]]:
        return [
            (None,) + input_shape_i if missing_bacthaxis_i else input_shape_i
            for input_shape_i, missing_bacthaxis_i in zip(input_shape[1:], self.missing_batchaxis[1:])
        ]
