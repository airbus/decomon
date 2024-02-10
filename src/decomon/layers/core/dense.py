import keras.ops as K
from keras.layers import Dense

from decomon.keras_utils import batch_multid_dot
from decomon.layers.layer import DecomonLayer
from decomon.types import Tensor


class DecomonNaiveDense(DecomonLayer):
    layer: Dense
    linear = True

    def get_affine_representation(self):
        w = self.layer.kernel
        b = self.layer.bias if self.layer.use_bias else K.zeros((self.layer.units,))

        # manage tensor-multid input
        for dim in self.layer.input.shape[-2:0:-1]:
            # Construct a multid-tensor diagonal by blocks
            reshaped_outer_shape = (dim, dim) + w.shape
            transposed_outer_axes = (
                (0,)
                + tuple(range(2, 2 + len(b.shape)))
                + (1,)
                + tuple(range(2 + len(b.shape), len(reshaped_outer_shape)))
            )
            w = K.transpose(K.reshape(K.outer(K.identity(dim), w), reshaped_outer_shape), transposed_outer_axes)
            # repeat bias along first dimensions
            b = K.repeat(b[None], dim, axis=0)

        return w, b


class DecomonDense(DecomonLayer):
    layer: Dense
    linear = True

    def _get_pseudo_affine_representation(self):
        w = self.layer.kernel
        b = self.layer.bias if self.layer.use_bias else K.zeros((self.layer.units,))
        return w, b

    def forward_ibp_propagate(self, lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
        w, b = self._get_pseudo_affine_representation()

        z_value = K.cast(0.0, dtype=w.dtype)
        w_pos = K.maximum(w, z_value)
        w_neg = K.minimum(w, z_value)

        kwargs_dot = dict(nb_merging_axes=1, missing_batchsize=(False, True))

        l_c = batch_multid_dot(lower, w_pos, **kwargs_dot) + batch_multid_dot(upper, w_neg, **kwargs_dot) + b
        u_c = batch_multid_dot(upper, w_pos, **kwargs_dot) + batch_multid_dot(lower, w_neg, **kwargs_dot) + b

        return l_c, u_c

    def forward_affine_propagate(
        self, input_affine_bounds, input_constant_bounds
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_l_1, b_l_1, w_u_1, b_u_1 = input_affine_bounds
        w_2, b_2 = self._get_pseudo_affine_representation()
        diagonal = (
            self.is_diagonal_bounds(input_affine_bounds),
            False,
        )
        missing_batchsize = (
            self.is_wo_batch_bounds(input_affine_bounds),
            True,
        )
        kwargs_dot_w = dict(nb_merging_axes=1, missing_batchsize=missing_batchsize, diagonal=diagonal)
        kwargs_dot_b = dict(nb_merging_axes=1, missing_batchsize=missing_batchsize)

        z_value = K.cast(0.0, dtype=w_2.dtype)
        w_2_pos = K.maximum(w_2, z_value)
        w_2_neg = K.minimum(w_2, z_value)

        w_l = batch_multid_dot(w_l_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_u_1, w_2_neg, **kwargs_dot_w)
        w_u = batch_multid_dot(w_u_1, w_2_pos, **kwargs_dot_w) + batch_multid_dot(w_l_1, w_2_neg, **kwargs_dot_w)
        b_l = batch_multid_dot(b_l_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_u_1, w_2_neg, **kwargs_dot_b) + b_2
        b_u = batch_multid_dot(b_u_1, w_2_pos, **kwargs_dot_b) + batch_multid_dot(b_l_1, w_2_neg, **kwargs_dot_b) + b_2

        return w_l, b_l, w_u, b_u

    def backward_affine_propagate(
        self, output_affine_bounds, input_constant_bounds
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        w_1, b_1 = self._get_pseudo_affine_representation()
        w_l_2, b_l_2, w_u_2, b_u_2 = output_affine_bounds

        # affine bounds represented in diagonal mode?
        diagonal_bounds = self.is_diagonal_bounds(output_affine_bounds)
        if diagonal_bounds:
            raise NotImplementedError
        # missing batch axis in affine bounds?
        nb_batch_axis = 0 if self.is_wo_batch_bounds(output_affine_bounds) else 1

        nb_nonbatch_axes_keras_input = len(w_l_2.shape) - len(b_l_2.shape)

        # Merge weights on "units" axis and reorder axes
        transposed_axes = (
            tuple(range(1, nb_nonbatch_axes_keras_input + nb_batch_axis))
            + (0,)
            + tuple(range(nb_nonbatch_axes_keras_input + nb_batch_axis, len(w_l_2.shape)))
        )
        w_l = K.transpose(
            K.tensordot(w_1, w_l_2, axes=[[-1], [nb_nonbatch_axes_keras_input - 1 + nb_batch_axis]]),
            axes=transposed_axes,
        )
        w_u = K.transpose(
            K.tensordot(w_1, w_u_2, axes=[[-1], [nb_nonbatch_axes_keras_input - 1 + nb_batch_axis]]),
            axes=transposed_axes,
        )

        # Merge layer bias with backward weights on "units" axe and reduce on other input axes
        reduced_axes = list(range(1, nb_nonbatch_axes_keras_input - 1 + nb_batch_axis))
        b_l = K.sum(
            K.tensordot(b_1, w_l_2, axes=[[-1], [nb_nonbatch_axes_keras_input - 1 + nb_batch_axis]]), axis=reduced_axes
        )
        b_u = K.sum(
            K.tensordot(b_1, w_u_2, axes=[[-1], [nb_nonbatch_axes_keras_input - 1 + nb_batch_axis]]), axis=reduced_axes
        )

        # Add bias from current backward bounds
        b_l += b_l_2
        b_u += b_u_2

        return w_l, b_l, w_u, b_u
