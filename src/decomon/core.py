from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Union

import keras.ops as K
import numpy as np
from keras.config import floatx

from decomon.keras_utils import batch_multid_dot
from decomon.types import Tensor


class Option(str, Enum):
    lagrangian = "lagrangian"
    milp = "milp"


class Slope(str, Enum):
    V_SLOPE = "volume-slope"
    A_SLOPE = "adaptative-slope"
    S_SLOPE = "same-slope"
    Z_SLOPE = "zero-lb"
    O_SLOPE = "one-lb"


class PerturbationDomain(ABC):
    opt_option: Option

    def __init__(self, opt_option: Union[str, Option] = Option.milp):
        self.opt_option = Option(opt_option)

    @abstractmethod
    def get_upper_x(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_lower_x(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        ...

    @abstractmethod
    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        ...

    @abstractmethod
    def get_nb_x_components(self) -> int:
        ...

    def get_config(self) -> dict[str, Any]:
        return {
            "opt_option": self.opt_option,
        }

    def get_x_input_shape_wo_batchsize(self, original_input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n_comp_x = self.get_nb_x_components()
        if n_comp_x == 1:
            return original_input_shape
        else:
            return (n_comp_x,) + original_input_shape


class BoxDomain(PerturbationDomain):
    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min=x_min, x_max=x_max, w=w, b=b, **kwargs)

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min=x_min, x_max=x_max, w=w, b=b, **kwargs)

    def get_upper_x(self, x: Tensor) -> Tensor:
        return x[:, 1]

    def get_lower_x(self, x: Tensor) -> Tensor:
        return x[:, 0]

    def get_nb_x_components(self) -> int:
        return 2


class GridDomain(PerturbationDomain):
    pass


class VertexDomain(PerturbationDomain):
    pass


class BallDomain(PerturbationDomain):
    def __init__(self, eps: float, p: float = 2, opt_option: Option = Option.milp):
        super().__init__(opt_option=opt_option)
        self.eps = eps
        # check on p
        p_error_msg = "p must be a positive integer or np.inf"
        try:
            if p != np.inf and (int(p) != p or p <= 0):
                raise ValueError(p_error_msg)
        except:
            raise ValueError(p_error_msg)
        self.p = p

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "eps": self.eps,
                "p": self.p,
            }
        )
        return config

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        return get_lower_ball(x_0=x, eps=self.eps, p=self.p, w=w, b=b, **kwargs)

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
        return get_upper_ball(x_0=x, eps=self.eps, p=self.p, w=w, b=b, **kwargs)

    def get_nb_x_components(self) -> int:
        return 1


class ForwardMode(str, Enum):
    """The different forward (from input to output) linear based relaxation perturbation analysis."""

    IBP = "ibp"
    """Propagation of constant bounds from input to output."""

    AFFINE = "affine"
    """Propagation of affine bounds from input to output."""

    HYBRID = "hybrid"
    """Propagation of constant and affines bounds from input to output."""


class Propagation(str, Enum):
    """Propagation direction."""

    FORWARD = "forward"
    BACKWARD = "backward"


def get_mode(ibp: bool = True, affine: bool = True) -> ForwardMode:
    if ibp:
        if affine:
            return ForwardMode.HYBRID
        else:
            return ForwardMode.IBP
    else:
        return ForwardMode.AFFINE


def get_ibp(mode: Union[str, ForwardMode] = ForwardMode.HYBRID) -> bool:
    mode = ForwardMode(mode)
    if mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
        return True
    return False


def get_affine(mode: Union[str, ForwardMode] = ForwardMode.HYBRID) -> bool:
    mode = ForwardMode(mode)
    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        return True
    return False


class InputsOutputsSpec:
    """Storing specifications for inputs and outputs of decomon/backward layer/model."""

    layer_input_shape: tuple[int, ...]
    model_input_shape: tuple[int, ...]
    model_output_shape: tuple[int, ...]

    def __init__(
        self,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        perturbation_domain: Optional[PerturbationDomain] = None,
        layer_input_shape: Optional[tuple[int, ...]] = None,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
    ):
        """
        Args:
            perturbation_domain: type of perturbation domain (box, ball, ...). Default to a box domain
            ibp: if True, forward propagate constant bounds
            affine: if True, forward propagate affine bounds
            propagation: direction of bounds propagation
              - forward: from input to output
              - backward: from output to input
            layer_input_shape: shape of the underlying keras layer input (w/o the batch axis)
            model_input_shape: shape of the underlying keras model input (w/o the batch axis)
            model_output_shape: shape of the underlying keras model output (w/o the batch axis)

        """
        # checks
        if propagation == Propagation.BACKWARD and model_output_shape is None:
            raise ValueError("model_output_shape must be set in backward propagation.")
        if propagation == Propagation.FORWARD and layer_input_shape is None:
            raise ValueError("layer_input_shape must be set in forward propagation.")

        self.propagation = propagation
        self.affine = affine
        self.ibp = ibp
        self.perturbation_domain: PerturbationDomain
        if perturbation_domain is None:
            self.perturbation_domain = BoxDomain()
        else:
            self.perturbation_domain = perturbation_domain
        if model_output_shape is None:
            self.model_output_shape = tuple()
        else:
            self.model_output_shape = model_output_shape
        if model_input_shape is None:
            self.model_input_shape = tuple()
        else:
            self.model_input_shape = model_input_shape
        if layer_input_shape is None:
            self.layer_input_shape = tuple()
        else:
            self.layer_input_shape = layer_input_shape

    def split_inputs(self, inputs: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        # Remove keras model input
        if self.propagation == Propagation.FORWARD and self.affine:
            x = inputs[-1]
            inputs = inputs[:-1]
            model_inputs = [x]
        else:
            model_inputs = []
        # Remove constant bounds
        if self.propagation == Propagation.BACKWARD or self.ibp:
            constant_oracle_bounds = inputs[-2:]
            inputs = inputs[:-2]
        else:
            constant_oracle_bounds = []
        # The remaining tensors are affine bounds
        # (potentially empty if: not backward or not affine or identity affine bounds)
        affine_bounds_to_propagate = inputs

        return affine_bounds_to_propagate, constant_oracle_bounds, model_inputs

    def split_input_shape(
        self, input_shape: list[tuple[Optional[int], ...]]
    ) -> tuple[list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]], list[tuple[Optional[int], ...]]]:
        return self.split_inputs(inputs=input_shape)  # type: ignore

    def flatten_inputs(
        self, affine_bounds_to_propagate: list[Tensor], constant_oracle_bounds: list[Tensor], model_inputs: list[Tensor]
    ) -> list[Tensor]:
        return affine_bounds_to_propagate + constant_oracle_bounds + model_inputs

    def split_outputs(self, outputs: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        # Remove constant bounds
        if self.propagation == Propagation.FORWARD and self.ibp:
            constant_bounds_propagated = outputs[-2:]
            outputs = outputs[:-2]
        else:
            constant_bounds_propagated = []
        # It remains affine bounds (can be empty if forward + not affine, or identity layer (e.g. DecomonLinear) on identity bounds
        affine_bounds_propagated = outputs

        return affine_bounds_propagated, constant_bounds_propagated

    def flatten_outputs(
        self, affine_bounds_propagated: list[Tensor], constant_bounds_propagated: Optional[list[Tensor]] = None
    ) -> list[Tensor]:
        if constant_bounds_propagated is None or self.propagation == Propagation.BACKWARD:
            return affine_bounds_propagated
        else:
            return affine_bounds_propagated + constant_bounds_propagated

    def flatten_outputs_shape(
        self,
        affine_bounds_propagated_shape: list[tuple[Optional[int], ...]],
        constant_bounds_propagated_shape: Optional[list[tuple[Optional[int], ...]]] = None,
    ) -> list[tuple[Optional[int], ...]]:
        return self.flatten_outputs(affine_bounds_propagated=affine_bounds_propagated_shape, constant_bounds_propagated=constant_bounds_propagated_shape)  # type: ignore

    def is_identity_bounds(self, affine_bounds: list[Tensor]) -> bool:
        return len(affine_bounds) == 0

    def is_identity_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        return len(affine_bounds_shape) == 0

    def is_diagonal_bounds(self, affine_bounds: list[Tensor]) -> bool:
        if self.is_identity_bounds(affine_bounds):
            return True
        w, b = affine_bounds[:2]
        return w.shape == b.shape

    def is_diagonal_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        if self.is_identity_bounds_shape(affine_bounds_shape):
            return True
        w_shape, b_shape = affine_bounds_shape[:2]
        return w_shape == b_shape

    def is_wo_batch_bounds(self, affine_bounds: list[Tensor]) -> bool:
        if self.is_identity_bounds(affine_bounds):
            return True
        b = affine_bounds[1]
        if self.propagation == Propagation.FORWARD:
            return len(b.shape) == len(self.layer_input_shape)
        else:
            return len(b.shape) == len(self.model_output_shape)

    def is_wo_batch_bounds_shape(self, affine_bounds_shape: list[tuple[Optional[int], ...]]) -> bool:
        if self.is_identity_bounds_shape(affine_bounds_shape):
            return True
        b_shape = affine_bounds_shape[1]
        if self.propagation == Propagation.FORWARD:
            return len(b_shape) == len(self.layer_input_shape)
        else:
            return len(b_shape) == len(self.model_output_shape)

    def get_kerasinputshape(self, inputsformode: list[Tensor]) -> tuple[Optional[int], ...]:
        return inputsformode[-1].shape

    def get_kerasinputshape_from_inputshapesformode(
        self, inputshapesformode: list[tuple[Optional[int], ...]]
    ) -> tuple[Optional[int], ...]:
        return inputshapesformode[-1]

    def get_fullinputshapes_from_inputshapesformode(
        self,
        inputshapesformode: list[tuple[Optional[int], ...]],
    ) -> list[tuple[Optional[int], ...]]:
        nb_tensors = self.nb_tensors
        empty_shape: tuple[Optional[int], ...] = tuple()
        if self.dc_decomp:
            if self.mode == ForwardMode.HYBRID:
                (
                    x_shape,
                    u_c_shape,
                    w_u_shape,
                    b_u_shape,
                    l_c_shape,
                    w_l_shape,
                    b_l_shape,
                    h_shape,
                    g_shape,
                ) = inputshapesformode[:nb_tensors]
            elif self.mode == ForwardMode.IBP:
                u_c_shape, l_c_shape, h_shape, g_shape = inputshapesformode[:nb_tensors]
                batchsize = u_c_shape[0]
                x_shape = (batchsize,) + self.perturbation_domain.get_x_input_shape_wo_batchsize(
                    (self.model_input_dim,)
                )
                b_shape = tuple(u_c_shape)
                w_shape = tuple(u_c_shape) + (u_c_shape[-1],)
                x_shape, w_u_shape, b_u_shape, w_l_shape, b_l_shape = (
                    x_shape,
                    w_shape,
                    b_shape,
                    w_shape,
                    b_shape,
                )
            elif self.mode == ForwardMode.AFFINE:
                x_shape, w_u_shape, b_u_shape, w_l_shape, b_l_shape, h_shape, g_shape = inputshapesformode[:nb_tensors]
                u_l_shape = tuple(b_u_shape)
                u_c_shape, l_c_shape = u_l_shape, u_l_shape
            else:
                raise ValueError(f"Unknown mode {self.mode}")
        else:
            h_shape, g_shape = empty_shape, empty_shape
            if self.mode == ForwardMode.HYBRID:
                x_shape, u_c_shape, w_u_shape, b_u_shape, l_c_shape, w_l_shape, b_l_shape = inputshapesformode[
                    :nb_tensors
                ]
            elif self.mode == ForwardMode.IBP:
                u_c_shape, l_c_shape = inputshapesformode[:nb_tensors]
                batchsize = u_c_shape[0]
                x_shape = (batchsize,) + self.perturbation_domain.get_x_input_shape_wo_batchsize(
                    (self.model_input_dim,)
                )
                b_shape = tuple(u_c_shape)
                w_shape = tuple(u_c_shape) + (u_c_shape[-1],)
                x_shape, w_u_shape, b_u_shape, w_l_shape, b_l_shape = (
                    x_shape,
                    w_shape,
                    b_shape,
                    w_shape,
                    b_shape,
                )
            elif self.mode == ForwardMode.AFFINE:
                x_shape, w_u_shape, b_u_shape, w_l_shape, b_l_shape = inputshapesformode[:nb_tensors]
                u_l_shape = tuple(b_u_shape)
                u_c_shape, l_c_shape = u_l_shape, u_l_shape
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        return [x_shape, u_c_shape, w_u_shape, b_u_shape, l_c_shape, w_l_shape, b_l_shape, h_shape, g_shape]

    def get_fullinputs_from_inputsformode(
        self, inputsformode: list[Tensor], compute_ibp_from_affine: bool = True, tight: bool = True
    ) -> list[Tensor]:
        """

        Args:
            inputsformode:
            compute_ibp_from_affine: if True and mode == affine, compute ibp bounds from affine ones
                with get_upper/get_lower
            tight: if True and mode==hybrid, compute tight ibp bounds, i.e. take tighter bound between
                - the ones from inputs, and
                - the ones computed from affine bounds with get_upper/get_lower

        Returns:

        """
        dtype = inputsformode[0].dtype
        nb_tensors = self.nb_tensors
        nonelike_tensor = self.get_empty_tensor(dtype=dtype)
        if self.dc_decomp:
            if self.mode == ForwardMode.HYBRID:
                x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputsformode[:nb_tensors]
            elif self.mode == ForwardMode.IBP:
                u_c, l_c, h, g = inputsformode[:nb_tensors]
                x, w_u, b_u, w_l, b_l = (
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                )
            elif self.mode == ForwardMode.AFFINE:
                x, w_u, b_u, w_l, b_l, h, g = inputsformode[:nb_tensors]
                u_c, l_c = nonelike_tensor, nonelike_tensor
            else:
                raise ValueError(f"Unknown mode {self.mode}")
        else:
            h, g = nonelike_tensor, nonelike_tensor
            if self.mode == ForwardMode.HYBRID:
                x, u_c, w_u, b_u, l_c, w_l, b_l = inputsformode[:nb_tensors]
            elif self.mode == ForwardMode.IBP:
                u_c, l_c = inputsformode[:nb_tensors]
                x, w_u, b_u, w_l, b_l = (
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                    nonelike_tensor,
                )
            elif self.mode == ForwardMode.AFFINE:
                x, w_u, b_u, w_l, b_l = inputsformode[:nb_tensors]
                u_c, l_c = nonelike_tensor, nonelike_tensor
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        compute_ibp_from_affine = (compute_ibp_from_affine and self.mode == ForwardMode.AFFINE) or (
            tight and self.mode == ForwardMode.HYBRID
        )

        if compute_ibp_from_affine:
            u_c_affine = self.perturbation_domain.get_upper(x, w_u, b_u)
            l_c_affine = self.perturbation_domain.get_lower(x, w_l, b_l)
            if self.mode == ForwardMode.AFFINE:
                u_c = u_c_affine
                l_c = l_c_affine
            else:
                u_c = K.minimum(u_c, u_c_affine)
                l_c = K.maximum(l_c, l_c_affine)

        return [x, u_c, w_u, b_u, l_c, w_l, b_l, h, g]

    def get_fullinputs_by_type_from_inputsformode_to_merge(
        self, inputsformode: list[Tensor], compute_ibp_from_affine: bool = False, tight: bool = True
    ) -> list[list[Tensor]]:
        """

        Args:
            inputsformode:
            compute_ibp_from_affine: if True and mode == affine, compute ibp bounds from affine ones
                with get_upper/get_lower
            tight: if True and mode==hybrid, compute tight ibp bounds, i.e. take tighter bound between
                - the ones from inputs, and
                - the ones computed from affine bounds with get_upper/get_lower

        Returns:

        """
        dtype = inputsformode[0].dtype
        nb_tensors_by_input = self.nb_tensors
        nb_inputs = len(inputsformode) // nb_tensors_by_input
        nonelike_tensor = self.get_empty_tensor(dtype=dtype)
        nonelike_tensor_list = [nonelike_tensor] * nb_inputs
        if self.mode == ForwardMode.HYBRID:
            inputs_x = inputsformode[0::nb_tensors_by_input]
            inputs_u_c = inputsformode[1::nb_tensors_by_input]
            inputs_w_u = inputsformode[2::nb_tensors_by_input]
            inputs_b_u = inputsformode[3::nb_tensors_by_input]
            inputs_l_c = inputsformode[4::nb_tensors_by_input]
            inputs_w_l = inputsformode[5::nb_tensors_by_input]
            inputs_b_l = inputsformode[6::nb_tensors_by_input]
        elif self.mode == ForwardMode.IBP:
            inputs_u_c = inputsformode[0::nb_tensors_by_input]
            inputs_l_c = inputsformode[1::nb_tensors_by_input]
            inputs_x, inputs_w_u, inputs_b_u, inputs_w_l, inputs_b_l = (
                nonelike_tensor_list,
                nonelike_tensor_list,
                nonelike_tensor_list,
                nonelike_tensor_list,
                nonelike_tensor_list,
            )
        elif self.mode == ForwardMode.AFFINE:
            inputs_x = inputsformode[0::nb_tensors_by_input]
            inputs_w_u = inputsformode[1::nb_tensors_by_input]
            inputs_b_u = inputsformode[2::nb_tensors_by_input]
            inputs_w_l = inputsformode[3::nb_tensors_by_input]
            inputs_b_l = inputsformode[4::nb_tensors_by_input]
            inputs_u_c, inputs_l_c = nonelike_tensor_list, nonelike_tensor_list
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.dc_decomp:
            inputs_h = inputsformode[nb_tensors_by_input - 2 :: nb_tensors_by_input]
            inputs_g = inputsformode[nb_tensors_by_input - 1 :: nb_tensors_by_input]
        else:
            inputs_h, inputs_g = nonelike_tensor_list, nonelike_tensor_list

        # compute ibp bounds from affine bounds
        compute_ibp_from_affine = (compute_ibp_from_affine and self.mode == ForwardMode.AFFINE) or (
            tight and self.mode == ForwardMode.HYBRID
        )

        if compute_ibp_from_affine:
            for i in range(len(inputs_x)):
                u_c_affine = self.perturbation_domain.get_upper(inputs_x[i], inputs_w_u[i], inputs_b_u[i])
                l_c_affine = self.perturbation_domain.get_lower(inputs_x[i], inputs_w_l[i], inputs_b_l[i])
                if self.mode == ForwardMode.AFFINE:
                    inputs_u_c[i] = u_c_affine
                    inputs_l_c[i] = l_c_affine
                else:
                    inputs_u_c[i] = K.minimum(inputs_u_c[i], u_c_affine)
                    inputs_l_c[i] = K.maximum(inputs_l_c[i], l_c_affine)

        return [inputs_x, inputs_u_c, inputs_w_u, inputs_b_u, inputs_l_c, inputs_w_l, inputs_b_l, inputs_h, inputs_g]

    def split_inputsformode_to_merge(self, inputsformode: list[Any]) -> list[list[Any]]:
        n_comp = self.nb_tensors
        return [inputsformode[n_comp * i : n_comp * (i + 1)] for i in range(len(inputsformode) // n_comp)]

    def extract_inputsformode_from_fullinputs(self, inputs: list[Tensor]) -> list[Tensor]:
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputs
        if self.mode == ForwardMode.HYBRID:
            inputsformode = [x, u_c, w_u, b_u, l_c, w_l, b_l]
        elif self.mode == ForwardMode.IBP:
            inputsformode = [u_c, l_c]
        elif self.mode == ForwardMode.AFFINE:
            inputsformode = [x, w_u, b_u, w_l, b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        if self.dc_decomp:
            inputsformode += [h, g]
        return inputsformode

    def extract_inputshapesformode_from_fullinputshapes(
        self, inputshapes: list[tuple[Optional[int], ...]]
    ) -> list[tuple[Optional[int], ...]]:
        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = inputshapes
        if self.mode == ForwardMode.HYBRID:
            inputshapesformode = [x, u_c, w_u, b_u, l_c, w_l, b_l]
        elif self.mode == ForwardMode.IBP:
            inputshapesformode = [u_c, l_c]
        elif self.mode == ForwardMode.AFFINE:
            inputshapesformode = [x, w_u, b_u, w_l, b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        if self.dc_decomp:
            inputshapesformode += [h, g]
        return inputshapesformode

    def extract_outputsformode_from_fulloutputs(self, outputs: list[Tensor]) -> list[Tensor]:
        return self.extract_inputsformode_from_fullinputs(outputs)

    @staticmethod
    def get_empty_tensor(dtype: Optional[str] = None) -> Tensor:
        if dtype is None:
            dtype = floatx()
        return K.convert_to_tensor([], dtype=dtype)


def get_upper_box(x_min: Tensor, x_max: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    """Compute the max of an affine function
    within a box (hypercube) defined by its extremal corners

    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        max_(x >= x_min, x<=x_max) w*x + b

    Note:
        We can have w, b in diagonal representation and/or without a batch axis.
        We assume that x_min, x_max have always its batch axis.

    """
    z_value = K.cast(0.0, dtype=x_min.dtype)
    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    is_diag = w.shape == b.shape
    is_wo_batch = len(b.shape) < len(x_min.shape)
    diagonal = (False, is_diag)
    missing_batchsize = (False, is_wo_batch)

    return (
        batch_multid_dot(x_max, w_pos, diagonal=diagonal, missing_batchsize=missing_batchsize)
        + batch_multid_dot(x_min, w_neg, diagonal=diagonal, missing_batchsize=missing_batchsize)
        + b
    )


def get_lower_box(x_min: Tensor, x_max: Tensor, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    """
    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w_l: weights of the affine lower bound
        b_l: bias of the affine lower bound

    Returns:
        min_(x >= x_min, x<=x_max) w*x + b

    Note:
        We can have w, b in diagonal representation and/or without a batch axis.
        We assume that x_min, x_max have always its batch axis.

    """
    return get_upper_box(x_min=x_max, x_max=x_min, w=w, b=b, **kwargs)


def get_lq_norm(x: Tensor, p: float, axis: int = -1) -> Tensor:
    """compute Lp norm (p=1 or 2)

    Args:
        x: tensor
        p: the power must be an integer in (1, 2)
        axis: the axis on which we compute the norm

    Returns:
        ||w||^p
    """
    if p == 1:
        x_q = K.max(K.abs(x), axis)
    elif p == 2:
        x_q = K.sqrt(K.sum(K.power(x, p), axis))
    else:
        raise NotImplementedError("p must be equal to 1 or 2")

    return x_q


def get_upper_ball(x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    """max of an affine function over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        max_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        raise NotImplementedError

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_upper_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        if len(kwargs):
            return get_upper_ball_finetune(x_0, eps, p, w, b, **kwargs)

        upper = eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + upper


def get_lower_ball(x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    """min of an affine fucntion over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        min_(|x - x_0|_p<= eps) w*x + b
    """
    if len(w.shape) == len(b.shape):
        return x_0 - eps

    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_lower_box(x_min, x_max, w, b)

    else:
        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        if len(kwargs):
            return get_lower_ball_finetune(x_0, eps, p, w, b, **kwargs)

        lower = -eps * get_lq_norm(w, p, axis=1) + b

        for _ in range(len(w.shape) - len(x_0.shape)):
            x_0 = K.expand_dims(x_0, -1)

        return K.sum(w * x_0, 1) + lower


def get_lower_ball_finetune(x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    if "finetune_lower" in kwargs and "upper" in kwargs or "lower" in kwargs:
        alpha = kwargs["finetune_lower"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2
        z_value = K.cast(0.0, dtype=w.dtype)

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(z_value, w_alpha_bar) * lower_reshaped, 1) + K.sum(
                K.minimum(z_value, w_alpha_bar) * upper_reshaped, 1
            )
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(z_value, w) * alpha[None] + K.maximum(z_value, w)
            w_alpha_bar = K.minimum(z_value, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(z_value, w_alpha_bar) * upper_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:
            lower = kwargs["lower"]  # flatten vector
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(z_value, w) * alpha[None] + K.minimum(z_value, w)
            w_alpha_bar = K.maximum(z_value, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(z_value, w_alpha_bar) * lower_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_lower_ball(x_0, eps, p, w, b)


def get_upper_ball_finetune(x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, **kwargs: Any) -> Tensor:
    if "finetune_upper" in kwargs and "upper" in kwargs or "lower" in kwargs:
        alpha = kwargs["finetune_upper"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2
        z_value = K.cast(0.0, dtype=w.dtype)

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(z_value, w_alpha_bar) * upper_reshaped, 1) + K.sum(
                K.minimum(z_value, w_alpha_bar) * lower_reshaped, 1
            )
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(z_value, w) * alpha[None] + K.maximum(z_value, w)
            w_alpha_bar = K.minimum(z_value, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(z_value, w_alpha_bar) * upper_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:
            lower = kwargs["lower"]  # flatten vector
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(z_value, w) * alpha[None] + K.minimum(z_value, w)
            w_alpha_bar = K.maximum(z_value, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(z_value, w_alpha_bar) * lower_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_upper_ball(x_0, eps, p, w, b)
