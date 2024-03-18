"""Convert decomon outputs to the specified format."""


from typing import Any, Optional

import keras.ops as K
from keras.layers import Layer
from keras.utils import serialize_keras_object

from decomon.layers.inputs_outputs_specs import InputsOutputsSpec
from decomon.layers.oracle import get_forward_oracle
from decomon.perturbation_domain import PerturbationDomain
from decomon.types import BackendTensor


class ConvertOutput(Layer):
    """Layer converting output of decomon model to desired final format."""

    def __init__(
        self,
        perturbation_domain: PerturbationDomain,
        ibp_from: bool,
        affine_from: bool,
        ibp_to: bool,
        affine_to: bool,
        model_output_shapes: list[tuple[int, ...]],
        **kwargs: Any,
    ):
        """
        Args:
            perturbation_domain: perturbation domain considered on keras model input
            ibp_from: ibp bounds present in current outputs?
            affine_from: affine bounds present in current outputs?
            ibp_to: ibp bounds to be present in new outputs?
            affine_to: affine bounds to be present in new outputs?
            model_output_shapes: shape of each output of the keras model (w/o batchsize)
            **kwargs: passed to Layer.__init__()

        """
        super().__init__(**kwargs)

        self.model_output_shapes = model_output_shapes
        self.affine_to = affine_to
        self.ibp_to = ibp_to
        self.affine_from = affine_from
        self.ibp_from = ibp_from
        self.perturbation_domain = perturbation_domain
        self.nb_outputs_keras_model = len(model_output_shapes)
        self.inputs_outputs_spec = InputsOutputsSpecForConvertOutput(
            ibp=ibp_from,
            affine=affine_from,
            ibp_to=ibp_to,
            affine_to=affine_to,
            model_output_shapes=model_output_shapes,
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ibp_from": self.ibp_from,
                "affine_from": self.affine_from,
                "ibp_to": self.ibp_to,
                "affine_to": self.affine_to,
                "model_output_shapes": self.model_output_shapes,
                "perturbation_domain": serialize_keras_object(self.perturbation_domain),
            }
        )
        return config

    def needs_perturbation_domain_inputs(self) -> bool:
        return self.inputs_outputs_spec.needs_perturbation_domain_inputs()

    def call(self, inputs: BackendTensor) -> list[BackendTensor]:
        """Compute ibp and affine bounds according to desired format from current decomon outputs.

        Args:
            inputs: sum_{i} (affine_bounds_from[i] + constant_bounds_from[i]) + perturbation_domain_inputs
                being the affine and constant bounds for each output of the keras model, with
                - i: the indice of the model output considered
                - sum_{i}: the concatenation of subsequent lists over i
                - affine_bounds_from[i]: empty if `self.affine_from` is False
                - constant_bounds_from[i]: empty if `self.ibp_from` is False
                - perturbation_domain_inputs: perturbation domain input wrapped in a list if self.ibp_from is False
                    and self.ibp_to is True, else empty

        Returns:
            sum_{i} (affine_bounds_to[i] + constant_bounds_to[i]): the affine and constant bounds computed
            - affine_bounds_to[i]: empty if `self.affine_to` is False
            - constant_bounds_to[i]: empty if `self.ibp_to` is False

        Computation:
        - ibp: if ibp_to is True and ibp_from is False, we use `decomon.layers.oracle.get_forward_oracle()`
        - affine: if affine_to is True and affine_from is False, we construct trivial bounds from ibp bounds
          w_l = w_u = 0 and b_l=lower, b_u=upper

        """
        affine_bounds_from, constant_bounds_from, perturbation_domain_inputs = self.inputs_outputs_spec.split_inputs(
            inputs
        )

        if self.ibp_to:
            if self.ibp_from:
                constant_bounds_to = constant_bounds_from
            else:
                from_linear = self.inputs_outputs_spec.is_wo_batch_bounds_by_keras_input(
                    affine_bounds=affine_bounds_from
                )
                constant_bounds_to = get_forward_oracle(
                    affine_bounds=affine_bounds_from,
                    ibp_bounds=constant_bounds_from,
                    perturbation_domain_inputs=perturbation_domain_inputs,
                    perturbation_domain=self.perturbation_domain,
                    ibp=self.ibp_from,
                    affine=self.affine_from,
                    is_merging_layer=True,
                    from_linear=from_linear,
                )
        else:
            constant_bounds_to = [[]] * self.nb_outputs_keras_model

        if self.affine_to:
            if self.affine_from:
                affine_bounds_to = affine_bounds_from
            else:
                x = perturbation_domain_inputs[0]
                keras_input_shape_with_batchsize = self.perturbation_domain.get_kerasinputlike_from_x(x).shape
                affine_bounds_to = []
                for constant_bounds_from_i in constant_bounds_from:
                    lower, upper = constant_bounds_from_i
                    w = K.zeros(keras_input_shape_with_batchsize + lower.shape[1:], dtype=x.dtype)
                    affine_bounds_to.append([w, lower, w, upper])
        else:
            affine_bounds_to = [[]] * self.nb_outputs_keras_model

        return self.inputs_outputs_spec.flatten_inputs(
            affine_bounds_to_propagate=affine_bounds_to,
            constant_oracle_bounds=constant_bounds_to,
            perturbation_domain_inputs=[],
        )

    def compute_output_shape(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        (
            affine_bounds_from_shape,
            constant_bounds_from_shape,
            perturbation_domain_inputs_shape,
        ) = self.inputs_outputs_spec.split_input_shape(input_shape)
        constant_bounds_to_shape: list[list[tuple[Optional[int], ...]]]
        affine_bounds_to_shape: list[list[tuple[Optional[int], ...]]]

        if self.ibp_to:
            if self.ibp_from:
                constant_bounds_to_shape = constant_bounds_from_shape
            else:
                constant_bounds_to_shape = []
                for model_output_shape in self.model_output_shapes:
                    lower_shape = (None,) + model_output_shape
                    constant_bounds_to_shape.append([lower_shape, lower_shape])
        else:
            constant_bounds_to_shape = [[]] * self.nb_outputs_keras_model

        if self.affine_to:
            if self.affine_from:
                affine_bounds_to_shape = affine_bounds_from_shape
            else:
                x_shape = perturbation_domain_inputs_shape[0]
                keras_input_shape = self.perturbation_domain.get_keras_input_shape_wo_batchsize(x_shape[1:])
                affine_bounds_to_shape = []
                for model_output_shape in self.model_output_shapes:
                    b_shape = (None,) + model_output_shape
                    w_shape = (None,) + keras_input_shape + model_output_shape
                    affine_bounds_to_shape.append([w_shape, b_shape, w_shape, b_shape])
        else:
            affine_bounds_to_shape = [[]] * self.nb_outputs_keras_model

        return self.inputs_outputs_spec.flatten_inputs_shape(
            affine_bounds_to_propagate_shape=affine_bounds_to_shape,
            constant_oracle_bounds_shape=constant_bounds_to_shape,
            perturbation_domain_inputs_shape=[],
        )


class InputsOutputsSpecForConvertOutput(InputsOutputsSpec):
    def __init__(
        self, ibp: bool, affine: bool, ibp_to: bool, affine_to: bool, model_output_shapes: list[tuple[int, ...]]
    ):
        super().__init__(ibp=ibp, affine=affine, layer_input_shape=model_output_shapes, is_merging_layer=True)
        self.affine_to = affine_to
        self.ibp_to = ibp_to

    def needs_perturbation_domain_inputs(self) -> bool:
        return (self.ibp_to and not self.ibp) or (self.affine_to and not self.affine)
