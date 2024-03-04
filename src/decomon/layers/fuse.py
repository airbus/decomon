"""Layers specifying constant oracle bounds on keras layer input."""


from typing import Any, Optional, Union, overload

import keras
from keras import ops as K
from keras.layers import Layer

from decomon.core import (
    BoxDomain,
    InputsOutputsSpec,
    PerturbationDomain,
    Propagation,
    get_lower_box,
    get_upper_box,
)
from decomon.keras_utils import add_tensors, batch_multid_dot
from decomon.types import BackendTensor, Tensor


class Fuse(Layer):
    """Layer combining bounds on successive models.

    We merge (ibp and/or affine) bounds on 2 models which are supposed to be chained.
    Both models can or not have both types of bounds

    We have, for any x in the given perturbation domain:

        m(x) = m2(m1(x))

        w_l_1 * x + b_l_1 <= m1(x) <= w_u_1 * x + b_u_1
        l_c_1 <= m1(x) <= u_c_1

        w_l_2 * m1(x) + b_l_2 <= m2(m1(x)) <= w_u_2 * m1(x) + b_u_2
        l_c_2 <= m2(m1(x)) <= u_c_2

    and we will deduce [w_l, b_l, w_u, b_u, u_c, l_c] such that

        w_l * x + b_l <= m(x) <= w_u * x + b_u
        l_c <= m(x) <= u_c

    In the case of multiple outputs for the first model, we assume that each output of m1 is branche to a model
    with a single output. That is to say, we have something like this:

        m(x)[i] = m2[i](m1(x)[i])

    i.e. i-th output of m(x) is the i-th output of m1(x) chained with a model m2[i] with single output.

    Hypothesis: the first model has only a single input.

    """

    def __init__(
        self,
        ibp_1: bool,
        affine_1: bool,
        ibp_2: bool,
        affine_2: bool,
        m1_input_shape: tuple[int, ...],
        m_1_output_shapes: list[tuple[int, ...]],
        from_linear_2: list[bool],
        **kwargs,
    ):
        """

        Args:
            ibp_1: specifying if first model constant bounds have been computed
            affine_1: specifying if first model affine bounds have been computed
            ibp_2: specifying if second model constant bounds have been computed
            affine_2: specifying if second model affine bounds have been computed
            m1_input_shape: input shape of the first model (w/o batchsize)
            m_1_output_shapes: shape of each output of the first model (w/o batchsize)
            from_linear_2: specifying if affine bounds for second model are from linear layers
                i.e. no bacthsize and w_l == w_u and b_l == b_u
            **kwargs: passed to Layer.__init__()

        """
        if not ibp_1 and not affine_1:
            raise ValueError("ibp_1 and affine_1 cannot be both False.")
        if not ibp_2 and not affine_2:
            raise ValueError("ibp_2 and affine_2 cannot be both False.")
        if len(m_1_output_shapes) == 0:
            raise ValueError("m_1_output_shapes cannot be empty")
        if not isinstance(m_1_output_shapes[0], tuple):
            raise ValueError("m_1_output_shapes must be a list of shapes (tuple of integers)")

        super().__init__(**kwargs)

        self.m_1_output_shapes = m_1_output_shapes
        self.m1_input_shape = m1_input_shape
        self.ibp_1 = ibp_1
        self.affine_1 = affine_1
        self.ibp_2 = ibp_2
        self.affine_2 = affine_2
        self.from_linear_2 = from_linear_2

        self.nb_outputs_first_model = len(m_1_output_shapes)

        self.ibp_fused = self.ibp_2 or (self.ibp_1 and self.affine_2)
        self.affine_fused = self.affine_1 and self.affine_2

        self.inputs_outputs_spec_1 = InputsOutputsSpec(ibp=ibp_1, affine=affine_1, layer_input_shape=m1_input_shape)
        self.inputs_outputs_spec_2 = [
            InputsOutputsSpec(ibp=ibp_2, affine=affine_2, layer_input_shape=m2_input_shape)
            for m2_input_shape in m_1_output_shapes
        ]

    def build(self, input_shape: tuple[list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]]]) -> None:
        input_shape_1, input_shape_2 = input_shape

        # check number of inputs
        if len(input_shape_1) != self.inputs_outputs_spec_1.nb_output_tensors * self.nb_outputs_first_model:
            raise ValueError(
                f"The first input should be a list whose length is the product of "
                f"{self.inputs_outputs_spec_1.nb_output_tensors} (number of tensors per bound) "
                f"by {self.nb_outputs_first_model} (number of outputs for the first model)."
            )

        if len(input_shape_2) != self.inputs_outputs_spec_2[0].nb_output_tensors * self.nb_outputs_first_model:
            raise ValueError(
                f"The second input should be a list whose length is the product of"
                f"{self.inputs_outputs_spec_2[0].nb_output_tensors} (number of tensors per bound) "
                f"by {self.nb_outputs_first_model} (number of outputs for the first model)."
            )

        self.built = True

    def _is_from_linear_m1_ith_affine_bounds(self, affine_bounds: list[Tensor], i: int) -> bool:
        return len(affine_bounds) == 0 or affine_bounds[1].shape == self.m_1_output_shapes[i]

    def _is_from_linear_m1_ith_affine_bounds_shape(
        self, affine_bounds_shape: list[tuple[Optional[int]]], i: int
    ) -> bool:
        return len(affine_bounds_shape) == 0 or affine_bounds_shape[1] == self.m_1_output_shapes[i]

    def call(self, inputs: tuple[list[BackendTensor], list[BackendTensor]]) -> list[BackendTensor]:
        """Fuse affine bounds.

        Args:
            inputs:  (sum_{i} affine_bounds_1[i] + constant_bounds_1[i], sum_{i} affine_bounds_2[i] + constant_bounds2[i])
                being the affine and constant bounds on first and second model for each output of the first model, with
                - i: the indice of the *first* model output considered
                - sum_{i}: the concatenation of subsequent lists over i
                - affine_bounds_1[i]: empty if `self.affine_1` is False
                - constant_bounds_1[i]: empty if `self.ibp_1` is False
                - affine_bounds_2[i]: empty if `self.affine_2` is False
                - constant_bounds_2[i]: empty if `self.ibp_2` is False

        Returns:
            sum_{i} affine_bounds_fused[i] + constant_bounds_fused[i]: fused affine and constant bounds for each output of the first model

        """
        bounds_1, bounds_2 = inputs

        bounds_fused: list[BackendTensor] = []
        for i in range(self.nb_outputs_first_model):
            bounds_1_i = bounds_1[
                i
                * self.inputs_outputs_spec_1.nb_output_tensors : (i + 1)
                * self.inputs_outputs_spec_1.nb_output_tensors
            ]
            affine_bounds_1, constant_bounds_1 = self.inputs_outputs_spec_1.split_outputs(bounds_1_i)

            bounds_2_i = bounds_2[
                i
                * self.inputs_outputs_spec_2[0].nb_output_tensors : (i + 1)
                * self.inputs_outputs_spec_2[0].nb_output_tensors
            ]
            affine_bounds_2, constant_bounds_2 = self.inputs_outputs_spec_2[0].split_outputs(bounds_2_i)

            # constant bounds
            if self.ibp_2:
                # ibp bounds already computed in second model
                constant_bounds_fused = constant_bounds_2
            elif self.ibp_1 and self.affine_2:
                # combine constant bounds on first model with affine bounds on second model
                lower, upper = constant_bounds_1
                constant_bounds_fused = list(
                    combine_affine_bound_with_constant_bound(
                        lower=lower,
                        upper=upper,
                        affine_bounds=affine_bounds_2,
                        missing_batchsize=self.from_linear_2[i],
                    )
                )
            else:
                constant_bounds_fused = []

            # affine bounds
            if self.affine_1 and self.affine_2:
                diagonal = (
                    self.inputs_outputs_spec_1.is_diagonal_bounds(affine_bounds_1),
                    self.inputs_outputs_spec_2[i].is_diagonal_bounds(affine_bounds_2),
                )
                from_linear_layer = (
                    self._is_from_linear_m1_ith_affine_bounds(affine_bounds=affine_bounds_1, i=i),
                    self.from_linear_2[i],
                )
                affine_bounds_fused = list(
                    combine_affine_bounds(
                        affine_bounds_1=affine_bounds_1,
                        affine_bounds_2=affine_bounds_2,
                        diagonal=diagonal,
                        from_linear_layer=from_linear_layer,
                    )
                )
            else:
                affine_bounds_fused = []

            # concatenate bounds
            bounds_fused += affine_bounds_fused + constant_bounds_fused

        return bounds_fused

    def compute_output_shape(
        self, input_shape: tuple[list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]]]
    ) -> list[tuple[Optional[int], ...]]:
        bounds_1_shape, bounds_2_shape = input_shape

        bounds_fused_shape: list[tuple[int, ...]] = []
        for i in range(self.nb_outputs_first_model):
            bounds_1_i_shape = bounds_1_shape[
                i
                * self.inputs_outputs_spec_1.nb_output_tensors : (i + 1)
                * self.inputs_outputs_spec_1.nb_output_tensors
            ]
            affine_bounds_1_shape, constant_bounds_1_shape = self.inputs_outputs_spec_1.split_output_shape(
                bounds_1_i_shape
            )

            bounds_2_i_shape = bounds_2_shape[
                i
                * self.inputs_outputs_spec_2[0].nb_output_tensors : (i + 1)
                * self.inputs_outputs_spec_2[0].nb_output_tensors
            ]
            affine_bounds_2_shape, constant_bounds_2_shape = self.inputs_outputs_spec_2[0].split_output_shape(
                bounds_2_i_shape
            )

            # constant bounds
            if self.ibp_2:
                # ibp bounds already computed in second model
                constant_bounds_fused_shape = constant_bounds_2_shape
            elif self.ibp_1 and self.affine_2:
                # combine constant bounds on first model with affine bounds on second model
                _, b2_shape, _, _ = affine_bounds_2_shape
                if self.from_linear_2[i]:
                    lower_fused_shape = (None,) + b2_shape
                else:
                    lower_fused_shape = b2_shape
                constant_bounds_fused_shape = [lower_fused_shape, lower_fused_shape]
            else:
                constant_bounds_fused_shape = []

            # affine bounds
            if self.affine_1 and self.affine_2:
                _, b2_shape, _, _ = affine_bounds_2_shape
                if self.from_linear_2[i]:
                    model_2_output_shape_wo_batchisze = b2_shape
                else:
                    model_2_output_shape_wo_batchisze = b2_shape[1:]

                diagonal = self.inputs_outputs_spec_1.is_diagonal_bounds_shape(
                    affine_bounds_1_shape
                ) and self.inputs_outputs_spec_2[i].is_diagonal_bounds_shape(affine_bounds_2_shape)
                if diagonal:
                    w_fused_shape_wo_batchsize = self.m1_input_shape
                else:
                    w_fused_shape_wo_batchsize = self.m1_input_shape + model_2_output_shape_wo_batchisze

                from_linear_layer = (
                    self._is_from_linear_m1_ith_affine_bounds_shape(affine_bounds_shape=affine_bounds_1_shape, i=i)
                    and self.from_linear_2[i]
                )
                if from_linear_layer:
                    w_fused_shape = w_fused_shape_wo_batchsize
                    b_fused_shape = model_2_output_shape_wo_batchisze
                else:
                    w_fused_shape = (None,) + w_fused_shape_wo_batchsize
                    b_fused_shape = (None,) + model_2_output_shape_wo_batchisze

                affine_bounds_fused_shape = [w_fused_shape, b_fused_shape, w_fused_shape, b_fused_shape]
            else:
                affine_bounds_fused_shape = []

            # concatenate bounds
            bounds_fused_shape += affine_bounds_fused_shape + constant_bounds_fused_shape

        return bounds_fused_shape


def combine_affine_bounds(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    from_linear_layer: tuple[bool, bool] = (False, False),
    diagonal: tuple[bool, bool] = (False, False),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        from_linear_layer: specify if affine_bounds_1 or affine_bounds_2
          come from the affine representation of a linear layer
        diagonal: specify if affine_bounds_1 or affine_bounds_2
          are in diagonal representation

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * y + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u


    Special cases

    - with linear layers:

        If the affine bounds come from the affine representation of a linear layer (e.g. affine_bounds_1), then
          - lower and upper bounds are equal: affine_bounds_1 = [w_1, b_1, w_1, b_1]
          - the tensors are missing the batch dimension

        In the generic case, tensors in affine_bounds have their first axis corresponding to the batch size.

    - diagonal representation:

        If w.shape == b.shape, this means that w is represented by its "diagonal" (potentially a tensor-multid).

    - empty affine bounds:

        when one affine bounds is an empty list, this is actually a convention for identity bounds, i.e.
          w = identity, b = 0
        therefore we return the other affine_bounds, unchanged.

    """
    # special case: empty bounds <=> identity bounds
    if len(affine_bounds_1) == 0:
        return tuple(affine_bounds_2)
    if len(affine_bounds_2) == 0:
        return tuple(affine_bounds_1)

    if from_linear_layer == (False, False):
        return _combine_affine_bounds_generic(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    elif from_linear_layer == (True, False):
        return _combine_affine_bounds_left_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    elif from_linear_layer == (False, True):
        return _combine_affine_bounds_right_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )
    else:
        return _combine_affine_bounds_both_from_linear(
            affine_bounds_1=affine_bounds_1, affine_bounds_2=affine_bounds_2, diagonal=diagonal
        )


def _combine_affine_bounds_generic(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * x + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_l_1, b_l_1, w_u_1, b_u_1 = affine_bounds_1
    w_l_2, b_l_2, w_u_2, b_u_2 = affine_bounds_2
    nb_axes_wo_batchsize_y = len(b_l_1.shape) - 1

    #  NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        diagonal=diagonal,
    )
    kwargs_dot_b: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        diagonal=(False, diagonal[1]),
    )

    z_value = K.cast(0.0, dtype=w_l_2.dtype)
    w_l_2_pos = K.maximum(w_l_2, z_value)
    w_u_2_pos = K.maximum(w_u_2, z_value)
    w_l_2_neg = K.minimum(w_l_2, z_value)
    w_u_2_neg = K.minimum(w_u_2, z_value)

    w_l = batch_multid_dot(w_l_1, w_l_2_pos, **kwargs_dot_w) + batch_multid_dot(w_u_1, w_l_2_neg, **kwargs_dot_w)
    w_u = batch_multid_dot(w_u_1, w_u_2_pos, **kwargs_dot_w) + batch_multid_dot(w_l_1, w_u_2_neg, **kwargs_dot_w)
    b_l = (
        batch_multid_dot(b_l_1, w_l_2_pos, **kwargs_dot_b) + batch_multid_dot(b_u_1, w_l_2_neg, **kwargs_dot_b) + b_l_2
    )
    b_u = (
        batch_multid_dot(b_u_1, w_u_2_pos, **kwargs_dot_b) + batch_multid_dot(b_l_1, w_u_2_neg, **kwargs_dot_b) + b_u_2
    )

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_right_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_l_1, b_l_1, w_u_1, b_u_1] first affine bounds
        affine_bounds_2: [w_2, b_2, w_2, b_2] second affine bounds, with lower=upper + no batchsize
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        w_l_1 * x + b_l_1 <= y <= w_u_1 * x + b_u_1
        z = w_2 * y + b_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_l_1, b_l_1, w_u_1, b_u_1 = affine_bounds_1
    w_2, b_2 = affine_bounds_2[:2]
    nb_axes_wo_batchsize_y = len(b_l_1.shape) - 1
    missing_batchsize = (False, True)

    #  NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    z_value = K.cast(0.0, dtype=w_2.dtype)
    w_2_pos = K.maximum(w_2, z_value)
    w_2_neg = K.minimum(w_2, z_value)

    w_l = batch_multid_dot(w_l_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_u_1, w_2_neg, **kwargs_dot_w)
    w_u = batch_multid_dot(w_u_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_l_1, w_2_neg, **kwargs_dot_w)
    b_l = batch_multid_dot(b_l_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_u_1, w_2_neg, **kwargs_dot_b) + b_2
    b_u = batch_multid_dot(b_u_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_l_1, w_2_neg, **kwargs_dot_b) + b_2

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_left_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_1, b_1, w_1, b_1] first affine bounds, with lower=upper + no batchsize
        affine_bounds_2: [w_l_2, b_l_2, w_u_2, b_u_2] second affine bounds
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w_l, b_l, w_u, b_u: combined affine bounds

    If x, y, z satisfy
        y = w_1 * x + b_1
        w_l_2 * y + b_l_2 <= z <= w_u_2 * x + b_u_2

    Then
        w_l * x + b_l <= z <= w_u * x + b_u

    """
    w_1, b_1 = affine_bounds_1[:2]
    w_l_2, b_l_2, w_u_2, b_u_2 = affine_bounds_2
    nb_axes_wo_batchsize_y = len(b_1.shape)
    missing_batchsize = (True, False)

    #   NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    w_l = batch_multid_dot(w_1, w_l_2, **kwargs_dot_w)
    w_u = batch_multid_dot(w_1, w_u_2, **kwargs_dot_w)
    b_l = batch_multid_dot(b_1, w_l_2, **kwargs_dot_b) + b_l_2
    b_u = batch_multid_dot(b_1, w_u_2, **kwargs_dot_b) + b_u_2

    return w_l, b_l, w_u, b_u


def _combine_affine_bounds_both_from_linear(
    affine_bounds_1: list[Tensor],
    affine_bounds_2: list[Tensor],
    diagonal: tuple[bool, bool],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Combine affine bounds

    Args:
        affine_bounds_1: [w_1, b_1, w_1, b_1] first affine bounds, with lower=upper + no batchsize
        affine_bounds_2: [w_2, b_2, w_2, b_2] second affine bounds, with lower=upper + no batchsize
        diagonal: specify if weights of each affine bounds are in diagonal representation or not

    Returns:
        w, b, w, b: combined affine bounds

    If x, y, z satisfy
        y = w_1 * x + b_1
        z = w_2 * x + b_2

    Then
        z = w * x + b

    """
    w_1, b_1 = affine_bounds_1[:2]
    w_2, b_2 = affine_bounds_2[:2]
    nb_axes_wo_batchsize_y = len(b_1.shape)
    missing_batchsize = (True, True)

    #   NB: bias is never a diagonal representation! => we split kwargs_dot_w and kwargs_dot_b
    kwargs_dot_w: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=diagonal,
    )
    kwargs_dot_b: dict[str, Any] = dict(
        nb_merging_axes=nb_axes_wo_batchsize_y,
        missing_batchsize=missing_batchsize,
        diagonal=(False, diagonal[1]),
    )

    w = batch_multid_dot(w_1, w_2, **kwargs_dot_w)
    b = batch_multid_dot(b_1, w_2, **kwargs_dot_b) + b_2

    return w, b, w, b


def combine_affine_bound_with_constant_bound(
    lower: Tensor,
    upper: Tensor,
    affine_bounds: list[Tensor],
    missing_batchsize: bool = False,
) -> tuple[Tensor, Tensor]:
    if len(affine_bounds) == 0:
        # identity affine bounds
        return lower, upper

    w_l, b_l, w_u, b_u = affine_bounds
    lower_fused = get_lower_box(x_min=lower, x_max=upper, w=w_l, b=b_l, missing_batchsize=missing_batchsize)
    upper_fused = get_upper_box(x_min=lower, x_max=upper, w=w_u, b=b_u, missing_batchsize=missing_batchsize)
    return lower_fused, upper_fused
