"""Generate decomon inputs from perturbation domain input."""


from typing import Any, Optional

import keras
import keras.ops as K
from keras.layers import Layer

from decomon.core import InputsOutputsSpec, PerturbationDomain, Propagation
from decomon.types import BackendTensor


class ForwardInput(Layer):
    """Layer generating the input of the first forward layer of a decomon layer."""

    def __init__(
        self,
        perturbation_domain: PerturbationDomain,
        ibp: bool,
        affine: bool,
        **kwargs: Any,
    ):
        """
        Args:
            perturbation_domain: default to a box domain
            ibp: if True, forward propagate constant bounds
            affine: if True, forward propagate affine bounds
            **kwargs:

        """
        super().__init__(**kwargs)

        self.perturbation_domain = perturbation_domain
        self.ibp = ibp
        self.affine = affine
        self.inputs_outputs_spec = InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            propagation=Propagation.FORWARD,
            perturbation_domain=perturbation_domain,
            model_input_shape=tuple(),
            layer_input_shape=tuple(),
        )

    def call(self, inputs: BackendTensor) -> list[BackendTensor]:
        """Generate ibp and affine bounds to propagate by the first forward layer.

        Args:
            inputs: the perturbation domain input

        Returns:
            affine_bounds + constant_bounds: the affine and constant bounds concatenated

            - affine_bounds: [w, 0, w, 0], with w the identity tensor of the proper shape
                (without batchsize, in diagonal representation)
            - constant_bounds: deduced from perturbation domain type and input

        """
        if self.affine:
            keras_input_like_tensor_wo_batchsize = self.perturbation_domain.get_kerasinputlike_from_x(x=inputs)[0]
            # identity: diag representation + w/o batchisze
            w = K.ones_like(keras_input_like_tensor_wo_batchsize)
            b = K.zeros_like(keras_input_like_tensor_wo_batchsize)
            affine_bounds = [w, b, w, b]
        else:
            affine_bounds = []
        if self.ibp:
            constant_bounds = [
                self.perturbation_domain.get_lower_x(x=inputs),
                self.perturbation_domain.get_upper_x(x=inputs),
            ]
        else:
            constant_bounds = []
        return self.inputs_outputs_spec.flatten_inputs(
            affine_bounds_to_propagate=affine_bounds,
            constant_oracle_bounds=constant_bounds,
            perturbation_domain_inputs=[],
        )

    def compute_output_shape(
        self,
        input_shape: tuple[Optional[int], ...],
    ) -> list[tuple[Optional[int], ...]]:
        perturbation_domain_input_shape_wo_batchsize = input_shape[1:]
        keras_input_shape_wo_batchsize = self.perturbation_domain.get_keras_input_shape_wo_batchsize(
            x_shape=perturbation_domain_input_shape_wo_batchsize
        )

        if self.affine:
            w_shape = keras_input_shape_wo_batchsize
            b_shape = keras_input_shape_wo_batchsize
            affine_bounds_shape = [w_shape, b_shape, w_shape, b_shape]
        else:
            affine_bounds_shape = []
        if self.ibp:
            lower_shape = (None,) + keras_input_shape_wo_batchsize
            constant_bounds_shape = [lower_shape, lower_shape]
        else:
            constant_bounds_shape = []
        return self.inputs_outputs_spec.flatten_inputs_shape(
            affine_bounds_to_propagate_shape=affine_bounds_shape,
            constant_oracle_bounds_shape=constant_bounds_shape,
            perturbation_domain_inputs_shape=[],
        )
