"""Converting backend tensors to symbolic tensors."""

from typing import Any, Optional

import keras
from keras.layers import Layer

from decomon.types import BackendTensor


class LinkToPerturbationDomainInput(Layer):
    """Layer making its inputs artificially depend on the first one.

    It accepts a list of tensors as its inputs. The convention is that:
    - The first tensor is the tensor we want the other to depend on, will be dropped in the output
    - The following tensors will be returned as is

    It returns the tensors, except the first one.

    The usecase is to be able to create a DecomonModel even when the propagated bounds does not really
    depend on the perturbation input.

    """

    def call(self, inputs: list[BackendTensor]) -> list[BackendTensor]:
        return inputs[1:]

    def compute_output_shape(self, input_shape: list[tuple[Optional[int], ...]]) -> list[tuple[Optional[int], ...]]:
        return input_shape[1:]
