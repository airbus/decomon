"""Layers specifying constant oracle bounds on keras layer input."""


from typing import Any, Optional, Union, overload

import keras
from keras.layers import Layer

from decomon.core import InputsOutputsSpec, PerturbationDomain
from decomon.types import BackendTensor


class BaseOracle(Layer):
    """Base class for oracle layers."""

    ...


class DecomonOracle(BaseOracle):
    """Layer deducing oracle bounds from decomon model outputs (forward or crown).

    The decomon model is supposed to give bounds for the inputs of the keras layer of interest.

    - In case of forward ibp or hybrid decomon model,
      the ibp bounds in decomon outputs are already the oracle bounds.
    - In case of forward pure affine or crown decomon model,
      the outputs are affine bounds on the inputs of the keras layer of interest w.r.t the keras model input.
      So we combine them with the perturbation domain input.

    Merging keras layer case: if the keras layer needing oracle is a merging layer,
    - the decomon model outputs are the concatenation of bounds on each keras layer input,
    - the oracle will be the concatenation of oracle bounds on each keras layer input

    """

    def __init__(
        self,
        perturbation_domain: PerturbationDomain,
        ibp: bool,
        affine: bool,
        layer_input_shape: Union[tuple[int, ...], list[tuple[int, ...]]],
        is_merging_layer: bool,
        **kwargs: Any,
    ):
        """
        Args:
            perturbation_domain: default to a box domain
            ibp: True for forward ibp or forward hybrid decomon model.
              In that case, the ibp bounds in the outputs are already the oracle bounds
            affine: True for forward affine (and hybrid) model and crown model.
              In that case (except for hybrid case, see above), the oracle bounds are deduce from the affine bounds.
            layer_input_shape: input shape of the underlying keras layer, without batchsize
            is_merging_layer: whether the underlying keras layer is a merging layer (i.e. with several inputs)
            **kwargs:

        """
        super().__init__(**kwargs)

        self.perturbation_domain = perturbation_domain
        self.ibp = ibp
        self.affine = affine
        self.is_merging_layer = is_merging_layer
        self.layer_input_shape = layer_input_shape
        self.inputs_outputs_spec = InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            perturbation_domain=perturbation_domain,
            layer_input_shape=layer_input_shape,
            is_merging_layer=is_merging_layer,
        )

    def call(self, inputs: list[BackendTensor]) -> Union[list[BackendTensor], list[list[BackendTensor]]]:
        """Deduce  ibp and affine bounds to propagate by the first forward layer.

        Args:
            inputs: the outputs of a decomon model + perturbation domain input if necessary

        Returns:
            concatenation of constant bounds on each keras layer input deduced from ibp and/or affine bounds

        According to the sense of propagation of the decomon model, we have

        - forward: inputs = affine_bounds + ibp_bounds + perturbation_domain_inputs with
            - affine_bounds: empty if self.affine = False
            - ibp_bounds: empty if self.ibp = False, already tight in hyb rid case (self.ibp=true and self.affine=True)
            - perturbation_domain_inputs: if self.ibp=False this is the pertubation domain input wrapped in a list, else empty.

            outputs = ibp_bounds when available, else combine affine_bounds with perturbation_domain_inputs

        - backward: inputs = crown_bounds + perturbation_domain_inputs with
            - affine_bounds: empty if self.affine = False
            - ibp_bounds: empty if self.ibp = False, already tight in hyb rid case (self.ibp=true and self.affine=True)
            - perturbation_domain_inputs: if self.ibp=False this is the pertubation domain input wrapped in a list, else empty.

            outputs = crown_bounds (affine) combined with perturbation domain input

        """
        (
            affine_bounds,
            ibp_bounds,
            perturbation_domain_inputs,
        ) = self.inputs_outputs_spec.split_inputs(inputs=inputs)

        return get_forward_oracle(
            affine_bounds=affine_bounds,
            ibp_bounds=ibp_bounds,
            perturbation_domain_inputs=perturbation_domain_inputs,
            perturbation_domain=self.perturbation_domain,
            ibp=self.ibp,
            affine=self.affine,
            is_merging_layer=self.is_merging_layer,
        )

    def compute_output_shape(
        self,
        input_shape: tuple[Optional[int], ...],
    ) -> Union[list[tuple[Optional[int], ...]], list[list[tuple[Optional[int], ...]]]]:
        """Compute output shape in case of symbolic call."""
        if self.is_merging_layer:
            output_shape = []
            for layer_input_shape_i in self.layer_input_shape:
                layer_input_shape_w_batchsize_i = (None,) + layer_input_shape_i
                output_shape.append([layer_input_shape_w_batchsize_i, layer_input_shape_w_batchsize_i])
            return output_shape
        else:
            layer_input_shape_w_batchsize = (None,) + self.layer_input_shape
            return [layer_input_shape_w_batchsize, layer_input_shape_w_batchsize]


@overload
def get_forward_oracle(
    affine_bounds: list[BackendTensor],
    ibp_bounds: list[BackendTensor],
    perturbation_domain_inputs: list[BackendTensor],
    perturbation_domain: PerturbationDomain,
    ibp: bool,
    affine: bool,
    is_merging_layer: bool,
) -> list[BackendTensor]:
    """Get constant oracle bounds on keras layer inputs from forward input bounds.

    Non-merging layer version.

    """
    ...


@overload
def get_forward_oracle(
    affine_bounds: list[list[BackendTensor]],
    ibp_bounds: list[list[BackendTensor]],
    perturbation_domain_inputs: list[BackendTensor],
    perturbation_domain: PerturbationDomain,
    ibp: bool,
    affine: bool,
    is_merging_layer: bool,
) -> list[list[BackendTensor]]:
    """Get constant oracle bounds on keras layer inputs from forward input bounds.

    Merging layer version.

    """
    ...


def get_forward_oracle(
    affine_bounds: Union[list[BackendTensor], list[list[BackendTensor]]],
    ibp_bounds: Union[list[BackendTensor], list[list[BackendTensor]]],
    perturbation_domain_inputs: list[BackendTensor],
    perturbation_domain: PerturbationDomain,
    ibp: bool,
    affine: bool,
    is_merging_layer: bool,
) -> Union[list[BackendTensor], list[list[BackendTensor]]]:
    """Get constant oracle bounds on keras layer inputs from forward input bounds.

    Args:
        affine_bounds: affine bounds on keras layer input w.r.t model input . Can be empty if not in affine mode.
        ibp_bounds: ibp constant bounds on keras layer input. Can be empty if not in ibp mode.
        perturbation_domain_inputs: perturbation domain input, wrapped in a list. Necessary only in affine mode, else empty.
        perturbation_domain: perturbation domain spec.
        ibp: ibp bounds exist?
        affine: affine bounds exist?
        is_merging_layer: keras layer is a merging layer?

    Returns:
        constant bounds on keras layer input deduced from forward layer input bounds or crown output + perturbation_domain_input

    In hybrid case (ibp+affine), the constant bounds are assumed to be already tight, which means the previous
    forward layer should already have took the tighter constant bounds between the ibp ones and the ones deduced
    from the affine bounds given the considered perturbation domain.

    """
    if ibp:
        # Hyp: in hybrid mode, the constant bounds are already tight
        # (affine and ibp mixed in forward layer output to get the tightest constant bounds)
        return ibp_bounds

    elif affine:
        if len(perturbation_domain_inputs) == 0:
            raise RuntimeError("Perturbation domain input is necessary for get_forward_oracle() in affine mode.")
        x = perturbation_domain_inputs[0]
        if is_merging_layer:
            constant_bounds = []
            for affine_bounds_i in affine_bounds:
                if len(affine_bounds_i) == 0:
                    # special case: empty affine bounds => identity bounds
                    l_affine = perturbation_domain.get_lower_x(x)
                    u_affine = perturbation_domain.get_upper_x(x)
                else:
                    w_l, b_l, w_u, b_u = affine_bounds_i
                    l_affine = perturbation_domain.get_lower(x, w_l, b_l)
                    u_affine = perturbation_domain.get_upper(x, w_u, b_u)
                constant_bounds.append([l_affine, u_affine])
            return constant_bounds
        else:
            if len(affine_bounds) == 0:
                # special case: empty affine bounds => identity bounds
                l_affine = perturbation_domain.get_lower_x(x)
                u_affine = perturbation_domain.get_upper_x(x)
            else:
                w_l, b_l, w_u, b_u = affine_bounds
                l_affine = perturbation_domain.get_lower(x, w_l, b_l)
                u_affine = perturbation_domain.get_upper(x, w_u, b_u)
            return [l_affine, u_affine]

    else:
        raise RuntimeError("ibp and affine cannot be both False")
