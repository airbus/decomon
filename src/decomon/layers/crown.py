"""Layers needed by crown algorithm."""


from typing import Any, Optional

import keras.ops as K
from keras.layers import Layer

from decomon.constants import Propagation
from decomon.keras_utils import add_tensors
from decomon.layers.inputs_outputs_specs import InputsOutputsSpec
from decomon.types import BackendTensor


class ReduceCrownBounds(Layer):
    """Layer reducing crown bounds computed from parent nodes of a merging layer.

    When encoutering a merging layer, crown algorithm propagates crown bounds for each parent.
    As keras models are supposed to have a single input, all resulting crown bounds are affine bounds
    on the same keras model output w.r.t this same single keras input.
    Thus we can reduce them to a single crown bound by simply summing each component (lower/upper weight/bias).

    NB: We need to take into account the fact that some bounds can be empty, diagonal, and/or w/o batchsize.

    """

    def __init__(
        self,
        model_output_shape: tuple[int, ...],
        **kwargs: Any,
    ):
        """
        Args:
            model_output_shape: shape of the model output for which the crown bounds have been computed
            **kwargs:

        """
        super().__init__(**kwargs)
        self.model_output_shape = model_output_shape

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "model_output_shape": self.model_output_shape,
            }
        )
        return config

    def call(self, inputs: list[list[BackendTensor]]) -> list[BackendTensor]:
        """Reduce the list of crown bounds to a single one by summation.

        Args:
            inputs: list of crown bounds of the form [[w_l[i], b_l[i], w_u[i], b_u[i]]_i]

        Returns:
            [w_l_tot, b_l_tot, w_u_tot, b_u_tot]: a single crown bound

        """
        if len(inputs) == 0:
            raise ValueError("inputs should not be empty")

        affine_bounds = inputs[0]
        for i in range(1, len(inputs)):
            affine_bounds_i = inputs[i]
            # identity: put on diag + w/o batchsize form to be able to sum
            missing_batchsize = (
                self.inputs_outputs_spec.is_wo_batch_bounds(affine_bounds, 0),
                self.inputs_outputs_spec.is_wo_batch_bounds(affine_bounds_i, i),
            )
            diagonal = (
                self.inputs_outputs_spec.is_diagonal_bounds(affine_bounds, 0),
                self.inputs_outputs_spec.is_diagonal_bounds(affine_bounds_i, i),
            )
            if len(affine_bounds) == 0:
                w = K.ones(self.model_output_shape)
                b = K.zeros(self.model_output_shape)
                w_l, b_l, w_u, b_u = w, b, w, b
            else:
                w_l, b_l, w_u, b_u = affine_bounds
            if len(affine_bounds_i) == 0:
                w = K.ones(self.model_output_shape)
                b = K.zeros(self.model_output_shape)
                w_l_i, b_l_i, w_u_i, b_u_i = w, b, w, b
            else:
                w_l_i, b_l_i, w_u_i, b_u_i = affine_bounds_i
            w_l = add_tensors(w_l, w_l_i, missing_batchsize=missing_batchsize, diagonal=diagonal)
            w_u = add_tensors(w_u, w_u_i, missing_batchsize=missing_batchsize, diagonal=diagonal)
            b_l = add_tensors(b_l, b_l_i, missing_batchsize=missing_batchsize)
            b_u = add_tensors(b_u, b_u_i, missing_batchsize=missing_batchsize)

            affine_bounds = [w_l, b_l, w_u, b_u]
        return affine_bounds

    def build(self, input_shape: list[list[tuple[Optional[int], ...]]]) -> None:
        self.nb_keras_inputs = len(input_shape)
        self.inputs_outputs_spec = InputsOutputsSpec(
            ibp=False,
            affine=True,
            propagation=Propagation.BACKWARD,
            is_merging_layer=True,
            model_output_shape=self.model_output_shape,
            layer_input_shape=[tuple()] * self.nb_keras_inputs,
        )
        self.built = True

    def compute_output_shape(
        self,
        input_shape: list[list[tuple[Optional[int], ...]]],
    ) -> list[tuple[Optional[int], ...]]:
        """Compute output shape in case of symbolic call."""
        if len(input_shape) == 0:
            raise ValueError("inputs should not be empty")
        return input_shape[0]
