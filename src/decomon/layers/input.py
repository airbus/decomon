"""Generate decomon inputs from perturbation domain input."""


from typing import Any, Optional, Union

import keras
import keras.ops as K
from keras.layers import Layer

from decomon.core import InputsOutputsSpec, PerturbationDomain, Propagation
from decomon.types import BackendTensor


class ForwardInput(Layer):
    """Layer generating the input of the first forward layer of a decomon model."""

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


class BackwardInput(Layer):
    """Layer preprocessing backward bounds to be used as input of the first backward layer of a decomon model.

    The backward bounds are supposed to be already flattened via `flatten_backward_bounds()`.
    This layer ensure having 4* nb_model_outputs tensors in backward_bounds so that it is the concatenation of
    backward bounds to be propagated for each keras model output.

    If this is already the case, this layer does nothing. Else there are 3 cases:

    - single tensor w: same bounds for each output, given by [w, 0, w, 0]
    - 2 tensors w, b: same bounds for each output, lower==upper, given by [w, b, w, b]
    - 4 tensors: same bounds for each output

    we concatenate the same bounds nb_model_outputs times.

    """

    single_tensor = False
    """A single tensor is given as entry."""
    lower_equal_upper = False
    """Lower and upper bounds are equal."""
    same_bounds_per_output = False
    """The same bounds are used for each keras model output."""

    def __init__(
        self,
        model_output_shapes: list[tuple[int, ...]],
        from_linear: list[bool],
        **kwargs: Any,
    ):
        """
        Args:
            model_output_shapes: shape of each output of the keras model (w/o batchsize)
            from_linear: specify if each backward bound corresponding to each keras model output is from a linear model
                (i.e. would be w/o batchsize and with lower == upper)
            **kwargs: passed to Layer.__init__()

        """
        super().__init__(**kwargs)

        self.model_output_shapes = model_output_shapes
        self.nb_model_outputs = len(model_output_shapes)
        self.from_linear = from_linear

    def build(self, input_shape: list[tuple[Optional[int], ...]]) -> None:
        # list of tensors
        if len(input_shape) == 1:
            # single tensor
            self.single_tensor = True
            self.lower_equal_upper = True
            self.same_bounds_per_output = True
        elif len(input_shape) == 2:
            # upper == lower, same for all model outputs
            self.lower_equal_upper = True
            self.same_bounds_per_output = True
        elif len(input_shape) == 4:
            # same bounds for all model outputs
            self.same_bounds_per_output = True
        elif len(input_shape) != 4 * self.nb_model_outputs:
            raise ValueError("backward_bounds should be given as a list of 1, 2, 4, or 4*nb_model_outputs tensors.")
        if self.same_bounds_per_output:
            for model_output_shape in self.model_output_shapes[1:]:
                if model_output_shape != self.model_output_shapes[0]:
                    raise ValueError(
                        "backward_bounds should be given for all model outputs "
                        "(i.e. as a list of 4*nb_model_outputs tensors) "
                        "if model outputs does not share the same shape"
                    )

        self.built = True

    def call(self, inputs: BackendTensor) -> list[BackendTensor]:
        """Reconstruct backward bounds at format needed by decomon conversion.

        Args:
            inputs: the flattened backward bounds given to `clone`

        Returns:
            backward_bounds as a list of 4 * nb_model_outputs tensors,
            concatenation of backward bounds on each keras model output

        """
        if self.same_bounds_per_output:
            if self.lower_equal_upper:
                if self.single_tensor:
                    w = inputs[0]
                    model_output_shape = self.model_output_shapes[0]
                    from_linear = self.from_linear[0]
                    if from_linear:
                        w_shape_wo_batchsize = w.shape
                    else:
                        w_shape_wo_batchsize = w.shape[1:]
                    is_diag = w_shape_wo_batchsize == model_output_shape
                    if is_diag:
                        m2_output_shape = model_output_shape
                    else:
                        m2_output_shape = w_shape_wo_batchsize[len(model_output_shape) :]
                    b_shape_wo_batchsize = m2_output_shape
                    if from_linear:
                        b_shape = b_shape_wo_batchsize
                    else:
                        batchsize = w.shape[0]
                        b_shape = (batchsize,) + b_shape_wo_batchsize
                    b = K.zeros(b_shape)
                else:
                    w, b = inputs
                bounds_per_output = [w, b, w, b]
            else:
                bounds_per_output = inputs
            return bounds_per_output * self.nb_model_outputs
        else:
            return inputs

    def compute_output_shape(
        self,
        input_shape: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        if self.same_bounds_per_output:
            if self.lower_equal_upper:
                if self.single_tensor:
                    w_shape = input_shape[0]
                    model_output_shape = self.model_output_shapes[0]
                    from_linear = self.from_linear[0]
                    if from_linear:
                        w_shape_wo_batchsize = w_shape
                    else:
                        w_shape_wo_batchsize = w_shape[1:]
                    is_diag = w_shape_wo_batchsize == model_output_shape
                    if is_diag:
                        m2_output_shape = model_output_shape
                    else:
                        m2_output_shape = w_shape_wo_batchsize[len(model_output_shape) :]
                    b_shape_wo_batchsize = m2_output_shape
                    if from_linear:
                        b_shape = b_shape_wo_batchsize
                    else:
                        batchsize = w_shape[0]
                        b_shape = (batchsize,) + b_shape_wo_batchsize
                else:
                    w_shape, b_shape = input_shape
                bounds_per_output_shape = [w_shape, b_shape, w_shape, b_shape]
            else:
                bounds_per_output_shape = input_shape
            return bounds_per_output_shape * self.nb_model_outputs
        else:
            return input_shape


def _is_keras_tensor_shape(shape):
    return len(shape) > 0 and (shape[0] is None or isinstance(shape[0], int))


def flatten_backward_bounds(
    backward_bounds: Union[keras.KerasTensor, list[keras.KerasTensor], list[list[keras.KerasTensor]]]
) -> list[keras.KerasTensor]:
    """Flatten backward bounds given to `clone`

    Args:
        backward_bounds:

    Returns:
        backward_bounds_flattened, computed from backward_bounds as follows:
        - single tensor -> [backward_bounds]
        - list of tensors -> backward_bounds
        - list of list of tensors -> flatten: [t for sublist in backward_bounds for t in sublist]

    """
    if isinstance(backward_bounds, keras.KerasTensor):
        return [backward_bounds]
    elif len(backward_bounds) == 0 or isinstance(backward_bounds[0], keras.KerasTensor):
        return backward_bounds
    else:
        return [t for sublist in backward_bounds for t in sublist]


def has_no_backward_bounds(
    backward_bounds: Optional[Union[keras.KerasTensor, list[keras.KerasTensor], list[list[keras.KerasTensor]]]],
) -> bool:
    """Check whether some backward bounds are to be propagated or not.

    Args:
        backward_bounds:

    Returns:

    """
    return backward_bounds is None or (
        not isinstance(backward_bounds, keras.KerasTensor)
        and (
            len(backward_bounds) == 0
            or all(
                not isinstance(backward_bounds_i, keras.KerasTensor) and len(backward_bounds_i) == 0
                for backward_bounds_i in backward_bounds
            )
        )
    )
