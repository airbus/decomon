from typing import Optional, Union, overload

from decomon.constants import Propagation
from decomon.types import Tensor


class InputsOutputsSpec:
    """Storing specifications for inputs and outputs of decomon/backward layer/model."""

    layer_input_shape: Union[tuple[int, ...], list[tuple[int, ...]]]
    model_input_shape: tuple[int, ...]
    model_output_shape: tuple[int, ...]

    def __init__(
        self,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        layer_input_shape: Optional[Union[tuple[int, ...], list[tuple[int, ...]]]] = None,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        is_merging_layer: bool = False,
        linear: bool = False,
    ):
        """
        Args:
            ibp: if True, forward propagate constant bounds
            affine: if True, forward propagate affine bounds
            propagation: direction of bounds propagation
              - forward: from input to output
              - backward: from output to input
            layer_input_shape: shape of the underlying keras layer input (w/o the batch axis)
            model_input_shape: shape of the underlying keras model input (w/o the batch axis)
            model_output_shape: shape of the underlying keras model output (w/o the batch axis)
            is_merging_layer: whether the underlying keras layer is a merging layer (i.e. with several inputs)
            linear: whether the underlying keras layer is linear (thus do not need oracle bounds for instance)

        """
        # checks
        if not ibp and not affine:
            raise ValueError("ibp and affine cannot be both False.")
        if propagation == Propagation.BACKWARD and model_output_shape is None:
            raise ValueError("model_output_shape must be set in backward propagation.")
        if propagation == Propagation.FORWARD or is_merging_layer:
            if layer_input_shape is None:
                raise ValueError("layer_input_shape must be set in forward propagation or for mergine layer.")
            elif is_merging_layer:
                if len(layer_input_shape) == 0 or not isinstance(layer_input_shape[0], tuple):
                    raise ValueError(
                        "layer_input_shape should be a non-empty list of shapes (tuple of int) for a merging layer."
                    )
            elif not isinstance(layer_input_shape, tuple) or (
                len(layer_input_shape) > 0 and not isinstance(layer_input_shape[0], int)
            ):
                raise ValueError("layer_input_shape should be a tuple of int for a unary layer.")

        self.propagation = propagation
        self.affine = affine
        self.ibp = ibp
        self.is_merging_layer = is_merging_layer
        self.linear = linear

        if model_output_shape is None:
            self.model_output_shape = tuple()
        else:
            self.model_output_shape = model_output_shape
        if model_input_shape is None:
            self.model_input_shape = tuple()
        else:
            self.model_input_shape = model_input_shape
        if layer_input_shape is None:
            if self.is_merging_layer:
                self.layer_input_shape = [tuple()]
            else:
                self.layer_input_shape = tuple()
        else:
            self.layer_input_shape = layer_input_shape

    def needs_perturbation_domain_inputs(self) -> bool:
        """Specify if decomon inputs should integrate keras model inputs."""
        return self.propagation == Propagation.FORWARD and self.affine

    def needs_oracle_bounds(self) -> bool:
        """Specify if decomon layer needs oracle bounds on keras layer inputs."""
        return not self.linear and (self.propagation == Propagation.BACKWARD or self.affine)

    def needs_constant_bounds_inputs(self) -> bool:
        """Specify if decomon inputs should integrate constant bounds."""
        return (self.propagation == Propagation.FORWARD and self.ibp) or (
            self.propagation == Propagation.BACKWARD and self.needs_oracle_bounds()
        )

    def needs_affine_bounds_inputs(self) -> bool:
        """Specify if decomon inputs should integrate affine bounds."""
        return (self.propagation == Propagation.FORWARD and self.affine) or (self.propagation == Propagation.BACKWARD)

    def cannot_have_empty_affine_inputs(self) -> bool:
        """Specify that it is not allowed to have empty affine bounds.

        Indeed, in merging case + forward propagation, it would be impossible to split decomon inputs properly.

        """
        return self.is_merging_layer and self.propagation == Propagation.FORWARD and self.affine

    @property
    def nb_keras_inputs(self) -> int:
        if self.is_merging_layer:
            return len(self.layer_input_shape)
        else:
            return 1

    @property
    def nb_input_tensors(self) -> int:
        nb = 0
        if self.propagation == Propagation.BACKWARD:
            # oracle bounds
            if self.needs_oracle_bounds():
                nb += 2 * self.nb_keras_inputs
            # affine
            nb += 4
            # model inputs
            if self.needs_perturbation_domain_inputs():
                nb += 1
        else:  # forward
            # ibp
            if self.ibp:
                nb += 2 * self.nb_keras_inputs
            # affine
            if self.affine:
                nb += 4 * self.nb_keras_inputs
            # model inputs
            if self.needs_perturbation_domain_inputs():
                nb += 1
        return nb

    @property
    def nb_output_tensors(self) -> int:
        nb = 0
        if self.propagation == Propagation.BACKWARD:
            nb += 4 * self.nb_keras_inputs
        else:  # forward
            if self.ibp:
                nb += 2
            if self.affine:
                nb += 4
        return nb

    @overload
    def split_constant_bounds(self, constant_bounds: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Split constant bounds, non-merging layer version."""
        ...

    @overload
    def split_constant_bounds(self, constant_bounds: list[list[Tensor]]) -> tuple[list[Tensor], list[Tensor]]:
        """Split constant bounds, merging layer version."""
        ...

    def split_constant_bounds(
        self, constant_bounds: Union[list[Tensor], list[list[Tensor]]]
    ) -> Union[tuple[Tensor, Tensor], tuple[list[Tensor], list[Tensor]]]:
        """Split constant bounds into lower, upper bound.

        Args:
            constant_bounds:
                if merging layer: list of constant (lower and upper) bounds for each keras layer inputs;
                else: list containing lower and upper bounds for the keras layer input.

        Returns:
            if merging_layer: 2 lists containing lower and upper bounds for each keras layer inputs;
            else: 2 tensors being the lower and upper bounds for the keras layer input.

        """
        if self.is_merging_layer:
            lowers, uppers = zip(*constant_bounds)
            return list(lowers), list(uppers)
        else:
            lower, upper = constant_bounds
            return lower, upper

    def split_inputs(
        self, inputs: list[Tensor]
    ) -> Union[
        tuple[list[Tensor], list[Tensor], list[Tensor]],
        tuple[list[list[Tensor]], list[list[Tensor]], list[Tensor]],
        tuple[list[Tensor], list[list[Tensor]], list[Tensor]],
    ]:
        """Split decomon inputs.

        Split them according to propagation mode and whether the underlying keras layer is merging or not.

        Args:
            inputs: flattened decomon inputs, as seen by `DecomonLayer.call()`.

        Returns:
            affine_bounds_to_propagate, constant_oracle_bounds, perturbation_domain_inputs:
                each one can be empty if not relevant,
                moreover, according to propagation mode and merging status,
                it will be list of tensors or list of lists of tensors.

        More details:

        - non-merging case:
            inputs = affine_bounds_to_propagate +  constant_oracle_bounds + perturbation_domain_inputs

        - merging case:
            - forward: k affine bounds to propagate w.r.t. each keras layer input + k constant bounds

                inputs = (
                    affine_bounds_to_propagate_0 +  constant_oracle_bounds_0 + ...
                    + affine_bounds_to_propagate_k +  constant_oracle_bounds_k
                    + perturbation_domain_inputs
                )

            - backward: only 1 affine bounds to propagate w.r.t keras layer output
                + k constant bounds w.r.t each keras layer input (empty if layer not linear)

                    inputs = (
                        affine_bounds_to_propagate
                        + constant_oracle_bounds_0 + ... +  constant_oracle_bounds_k
                        + perturbation_domain_inputs
                    )
        Note: in case of merging layer + forward, we should not have empty affine bounds
          as it will be impossible to split properly the inputs.

        """
        # Remove keras model input
        if self.needs_perturbation_domain_inputs():
            x = inputs[-1]
            inputs = inputs[:-1]
            perturbation_domain_inputs = [x]
        else:
            perturbation_domain_inputs = []
        if self.is_merging_layer:
            if self.propagation == Propagation.BACKWARD:
                # expected number of constant bounds
                nb_constant_bounds_by_keras_input = 2 if self.needs_oracle_bounds() else 0
                nb_constant_bounds = self.nb_keras_inputs * nb_constant_bounds_by_keras_input
                # remove affine bounds (could be empty to express identity bounds)
                affine_bounds_to_propagate = inputs[: len(inputs) - nb_constant_bounds]
                inputs = inputs[len(inputs) - nb_constant_bounds :]
                # split constant bounds by keras input
                if nb_constant_bounds > 0:
                    constant_oracle_bounds = [
                        [inputs[i], inputs[i + 1]] for i in range(0, len(inputs), nb_constant_bounds_by_keras_input)
                    ]
                else:
                    constant_oracle_bounds = []
            else:  # forward
                # split bounds by keras input
                nb_affine_bounds_by_keras_input = 4 if self.affine else 0
                nb_constant_bounds_by_keras_input = 2 if self.ibp else 0
                nb_bounds_by_keras_input = nb_affine_bounds_by_keras_input + nb_constant_bounds_by_keras_input
                affine_bounds_to_propagate = [
                    [inputs[start_input + j_bound] for j_bound in range(nb_affine_bounds_by_keras_input)]
                    for start_input in range(0, len(inputs), nb_bounds_by_keras_input)
                ]
                constant_oracle_bounds = [
                    [
                        inputs[start_input + nb_affine_bounds_by_keras_input + j_bound]
                        for j_bound in range(nb_constant_bounds_by_keras_input)
                    ]
                    for start_input in range(0, len(inputs), nb_bounds_by_keras_input)
                ]
        else:
            # Remove constant bounds
            if self.needs_constant_bounds_inputs():
                constant_oracle_bounds = inputs[-2:]
                inputs = inputs[:-2]
            else:
                constant_oracle_bounds = []
            # The remaining tensors are affine bounds
            # (potentially empty if: not backward or not affine or identity affine bounds)
            affine_bounds_to_propagate = inputs

        return affine_bounds_to_propagate, constant_oracle_bounds, perturbation_domain_inputs

    def split_input_shape(
        self, input_shape: list[tuple[Optional[int], ...]]
    ) -> Union[
        tuple[list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]]],
        tuple[
            list[list[tuple[Optional[int], ...]]],
            list[list[tuple[Optional[int], ...]]],
            list[tuple[Optional[int], ...]],
        ],
        tuple[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]], list[tuple[Optional[int], ...]]],
    ]:
        """Split decomon inputs.

        Split them according to propagation mode and whether the underlying keras layer is merging or not.

        Args:
            input_shape: flattened decomon inputs, as seen by `DecomonLayer.call()`.

        Returns:
            affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, perturbation_domain_inputs_shape:
                each one can be empty if not relevant, and according to propagation mode and merging status,
                it will be list of shapes or list of lists of shapes.

        """
        return self.split_inputs(inputs=input_shape)  # type: ignore

    def flatten_inputs(
        self,
        affine_bounds_to_propagate: Union[list[Tensor], list[list[Tensor]]],
        constant_oracle_bounds: Union[list[Tensor], list[list[Tensor]]],
        perturbation_domain_inputs: list[Tensor],
    ) -> list[Tensor]:
        """Flatten decomon inputs.

        Reverse `self.split_inputs()`.

        Args:
            affine_bounds_to_propagate:
                - forward + affine: affine bounds on each keras layer input w.r.t. model input
                    -> list of lists of tensors in merging case;
                    -> list of tensors else.
                - backward: affine bounds on model output w.r.t keras layer output
                    -> list of tensors
                - else: empty
            constant_oracle_bounds:
                - forward + ibp: ibp bounds on keras layer inputs
                - backward + not linear: oracle bounds on keras layer inputs
                - else: empty
            perturbation_domain_inputs:
                - forward + affine: perturbation domain input wrapped in a list
                - else: empty

        Returns:
            flattened inputs
            - non-merging case:
                inputs = affine_bounds_to_propagate +  constant_oracle_bounds + perturbation_domain_inputs

            - merging case:
                - forward: k affine bounds to propagate w.r.t. each keras layer input + k constant bounds

                    inputs = (
                        affine_bounds_to_propagate_0 +  constant_oracle_bounds_0 + ...
                        + affine_bounds_to_propagate_k +  constant_oracle_bounds_k
                        + perturbation_domain_inputs
                    )

                - backward: only 1 affine bounds to propagate w.r.t keras layer output
                    + k constant bounds w.r.t each keras layer input  (empty of linear layer)

                    inputs = (
                        affine_bounds_to_propagate
                        + constant_oracle_bounds_0 + ... +  constant_oracle_bounds_k
                        + perturbation_domain_inputs
                    )

        """
        if self.is_merging_layer:
            if self.propagation == Propagation.BACKWARD:
                if self.needs_oracle_bounds():
                    flattened_constant_oracle_bounds = [
                        t for constant_oracle_bounds_i in constant_oracle_bounds for t in constant_oracle_bounds_i
                    ]
                else:
                    flattened_constant_oracle_bounds = []
                return affine_bounds_to_propagate + flattened_constant_oracle_bounds + perturbation_domain_inputs
            else:  # forward
                bounds_by_keras_input = [
                    affine_bounds_to_propagate_i + constant_oracle_bounds_i
                    for affine_bounds_to_propagate_i, constant_oracle_bounds_i in zip(
                        affine_bounds_to_propagate, constant_oracle_bounds
                    )
                ]
                flattened_bounds_by_keras_input = [
                    t for bounds_by_keras_input_i in bounds_by_keras_input for t in bounds_by_keras_input_i
                ]
                return flattened_bounds_by_keras_input + perturbation_domain_inputs
        else:
            return affine_bounds_to_propagate + constant_oracle_bounds + perturbation_domain_inputs

    def flatten_inputs_shape(
        self,
        affine_bounds_to_propagate_shape: Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]],
        constant_oracle_bounds_shape: Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]],
        perturbation_domain_inputs_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        """Flatten inputs shape

        Same operation as `flatten_inputs` but on tensor shapes.

        Args:
            affine_bounds_to_propagate_shape:
            constant_oracle_bounds_shape:
            perturbation_domain_inputs_shape:

        Returns:

        """
        return self.flatten_inputs(  # type: ignore
            affine_bounds_to_propagate=affine_bounds_to_propagate_shape,
            constant_oracle_bounds=constant_oracle_bounds_shape,
            perturbation_domain_inputs=perturbation_domain_inputs_shape,
        )  # type: ignore

    def split_outputs(self, outputs: list[Tensor]) -> tuple[Union[list[Tensor], list[list[Tensor]]], list[Tensor]]:
        """Split decomon inputs.

        Reverse operation of `self.flatten_outputs()`

        Args:
            outputs: flattened decomon outputs, as returned by `DecomonLayer.call()`.

        Returns:
            affine_bounds_propagated, constant_bounds_propagated:
                each one can be empty if not relevant and can be list of tensors or a list of lists of tensors
                according to propagation and merging status.

        More details:

            - forward: affine_bounds_propagated, constant_bounds_propagated: both simple lists of tensors corresponding to
                affine and constant bounds on keras layer output.
            - backward: constant_bounds_propagated is empty (not relevant) and
                - merging layer: affine_bounds_propagated is a list of lists of tensors corresponding
                    to partial affine bounds on model output w.r.t each keras input
                - else: affine_bounds_propagated is a simple list of tensors

        """
        # Remove constant bounds
        if self.propagation == Propagation.FORWARD and self.ibp:
            constant_bounds_propagated = outputs[-2:]
            outputs = outputs[:-2]
        else:
            constant_bounds_propagated = []
        # It remains affine bounds (can be empty if forward + not affine, or identity layer (e.g. DecomonLinear) on identity bounds
        affine_bounds_propagated = outputs
        if self.propagation == Propagation.BACKWARD and self.is_merging_layer:
            nb_affine_bounds_by_keras_input = 4
            affine_bounds_propagated = [
                affine_bounds_propagated[i : i + nb_affine_bounds_by_keras_input]
                for i in range(0, len(affine_bounds_propagated), nb_affine_bounds_by_keras_input)
            ]

        return affine_bounds_propagated, constant_bounds_propagated

    def split_output_shape(
        self, output_shape: list[tuple[Optional[int], ...]]
    ) -> tuple[
        Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]], list[tuple[Optional[int], ...]]
    ]:
        """Split decomon output shape."""
        return self.split_outputs(outputs=output_shape)  # type: ignore

    def flatten_outputs(
        self,
        affine_bounds_propagated: Union[list[Tensor], list[list[Tensor]]],
        constant_bounds_propagated: Optional[list[Tensor]] = None,
    ) -> list[Tensor]:
        """Flatten decomon outputs.

        Args:
            affine_bounds_propagated:
                - forward + affine: affine bounds on keras layer output w.r.t. model input
                - backward: affine bounds on model output w.r.t each keras layer input
                  -> list of lists of tensors in merging case;
                  -> list of tensors else.
                - else: empty
            constant_bounds_propagated:
                - forward + ibp: ibp bounds on keras layer output
                - else: empty or None

        Returns:
            flattened outputs
            - forward: affine_bounds_propagated + constant_bounds_propagated
            - backward:
                - merging layer (k keras layer inputs): affine_bounds_propagated_0 + ... + affine_bounds_propagated_k
                - else: affine_bounds_propagated

        """
        if constant_bounds_propagated is None or self.propagation == Propagation.BACKWARD:
            if self.is_merging_layer and self.propagation == Propagation.BACKWARD:
                return [
                    t for affine_bounds_propagated_i in affine_bounds_propagated for t in affine_bounds_propagated_i
                ]
            else:
                return affine_bounds_propagated
        else:
            return affine_bounds_propagated + constant_bounds_propagated

    def flatten_outputs_shape(
        self,
        affine_bounds_propagated_shape: Union[
            list[tuple[Optional[int], ...]],
            list[list[tuple[Optional[int], ...]]],
        ],
        constant_bounds_propagated_shape: Optional[list[tuple[Optional[int], ...]]] = None,
    ) -> list[tuple[Optional[int], ...]]:
        """Flatten decomon outputs shape."""
        return self.flatten_outputs(affine_bounds_propagated=affine_bounds_propagated_shape, constant_bounds_propagated=constant_bounds_propagated_shape)  # type: ignore

    def has_multiple_bounds_inputs(self) -> bool:
        return self.propagation == Propagation.FORWARD and self.is_merging_layer

    @overload
    def extract_shapes_from_affine_bounds(
        self, affine_bounds: list[Tensor], i: int = -1
    ) -> list[tuple[Optional[int], ...]]:
        ...

    @overload
    def extract_shapes_from_affine_bounds(
        self, affine_bounds: list[list[Tensor]], i: int = -1
    ) -> list[list[tuple[Optional[int], ...]]]:
        ...

    def extract_shapes_from_affine_bounds(
        self, affine_bounds: Union[list[Tensor], list[list[Tensor]]], i: int = -1
    ) -> Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]]:
        if self.has_multiple_bounds_inputs() and i == -1:
            return [[t.shape for t in sub_bounds] for sub_bounds in affine_bounds]
        else:
            return [t.shape for t in affine_bounds]  # type: ignore

    def is_identity_bounds(self, affine_bounds: Union[list[Tensor], list[list[Tensor]]], i: int = -1) -> bool:
        return self.is_identity_bounds_shape(
            affine_bounds_shape=self.extract_shapes_from_affine_bounds(affine_bounds=affine_bounds, i=i), i=i
        )

    def is_identity_bounds_shape(
        self,
        affine_bounds_shape: Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]],
        i: int = -1,
    ) -> bool:
        if self.has_multiple_bounds_inputs() and i == -1:
            return all(
                self.is_identity_bounds_shape(affine_bounds_shape=affine_bounds_shape[i], i=i)  # type: ignore
                for i in range(self.nb_keras_inputs)
            )
        else:
            return len(affine_bounds_shape) == 0

    def is_diagonal_bounds(self, affine_bounds: Union[list[Tensor], list[list[Tensor]]], i: int = -1) -> bool:
        return self.is_diagonal_bounds_shape(
            affine_bounds_shape=self.extract_shapes_from_affine_bounds(affine_bounds=affine_bounds, i=i), i=i
        )

    def is_diagonal_bounds_shape(
        self,
        affine_bounds_shape: Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]],
        i: int = -1,
    ) -> bool:
        if self.has_multiple_bounds_inputs() and i == -1:
            return all(
                self.is_diagonal_bounds_shape(affine_bounds_shape=affine_bounds_shape[i], i=i)  # type: ignore
                for i in range(self.nb_keras_inputs)
            )
        else:
            if self.is_identity_bounds_shape(affine_bounds_shape, i=i):
                return True
            w_shape, b_shape = affine_bounds_shape[:2]
            return w_shape == b_shape

    def is_wo_batch_bounds(self, affine_bounds: Union[list[Tensor], list[list[Tensor]]], i: int = -1) -> bool:
        return self.is_wo_batch_bounds_shape(
            affine_bounds_shape=self.extract_shapes_from_affine_bounds(affine_bounds=affine_bounds, i=i), i=i
        )

    def is_wo_batch_bounds_shape(
        self,
        affine_bounds_shape: Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]],
        i: int = -1,
    ) -> bool:
        if self.has_multiple_bounds_inputs() and i == -1:
            return all(
                self.is_wo_batch_bounds_shape(affine_bounds_shape=affine_bounds_shape[i], i=i)  # type: ignore
                for i in range(self.nb_keras_inputs)
            )
        else:
            if self.is_identity_bounds_shape(affine_bounds_shape, i=i):
                return True
            b_shape = affine_bounds_shape[1]
            if self.propagation == Propagation.FORWARD:
                if i > -1:
                    return len(b_shape) == len(self.layer_input_shape[i])
                else:
                    return len(b_shape) == len(self.layer_input_shape)
            else:
                return len(b_shape) == len(self.model_output_shape)

    @overload
    def is_wo_batch_bounds_by_keras_input(
        self,
        affine_bounds: list[Tensor],
    ) -> bool:
        ...

    @overload
    def is_wo_batch_bounds_by_keras_input(
        self,
        affine_bounds: list[list[Tensor]],
    ) -> list[bool]:
        ...

    def is_wo_batch_bounds_by_keras_input(
        self,
        affine_bounds: Union[list[Tensor], list[list[Tensor]]],
    ) -> Union[bool, list[bool]]:
        if self.has_multiple_bounds_inputs():
            return [self.is_wo_batch_bounds(affine_bounds_i, i=i) for i, affine_bounds_i in enumerate(affine_bounds)]
        else:
            return self.is_wo_batch_bounds(affine_bounds)
