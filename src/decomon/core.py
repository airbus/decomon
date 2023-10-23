from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import keras.ops as K
import numpy as np
from keras.config import floatx


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
    def get_upper(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
        ...

    @abstractmethod
    def get_lower(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
        ...

    @abstractmethod
    def get_nb_x_components(self) -> int:
        ...

    def get_config(self) -> Dict[str, Any]:
        return {
            "opt_option": self.opt_option,
        }

    def get_x_input_shape(self, original_input_dim: int) -> Tuple[int, ...]:
        n_comp_x = self.get_nb_x_components()
        if n_comp_x == 1:
            return (original_input_dim,)
        else:
            return (
                n_comp_x,
                original_input_dim,
            )


class BoxDomain(PerturbationDomain):
    def get_upper(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min=x_min, x_max=x_max, w=w, b=b, **kwargs)

    def get_lower(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min=x_min, x_max=x_max, w=w, b=b, **kwargs)

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

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "eps": self.eps,
                "p": self.p,
            }
        )
        return config

    def get_lower(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
        return get_lower_ball(x_0=x, eps=self.eps, p=self.p, w=w, b=b, **kwargs)

    def get_upper(
        self, x: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
    ) -> keras.KerasTensor:
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

    def __init__(
        self,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
    ):
        """
        Args:
            dc_decomp: boolean that indicates whether we return a
                difference of convex decomposition of our layer
            mode: type of Forward propagation (ibp, affine, or hybrid)
            perturbation_domain: type of perturbation domain (box, ball, ...)

        """

        self.mode = ForwardMode(mode)
        self.dc_decomp = dc_decomp
        self.perturbation_domain: PerturbationDomain
        if perturbation_domain is None:
            self.perturbation_domain = BoxDomain()
        else:
            self.perturbation_domain = perturbation_domain

    @property
    def nb_tensors(self) -> int:
        if self.mode == ForwardMode.HYBRID:
            nb_tensors = 7
        elif self.mode == ForwardMode.IBP:
            nb_tensors = 2
        elif self.mode == ForwardMode.AFFINE:
            nb_tensors = 5
        else:
            raise NotImplementedError(f"unknown forward mode {self.mode}")

        if self.dc_decomp:
            nb_tensors += 2

        return nb_tensors

    @property
    def ibp(self) -> bool:
        return get_ibp(self.mode)

    @property
    def affine(self) -> bool:
        return get_affine(self.mode)

    def get_input_shape(self, inputsformode: List[keras.KerasTensor]) -> Tuple[Optional[int]]:
        return inputsformode[-1].shape

    def get_fullinputs_from_inputsformode(
        self, inputsformode: List[keras.KerasTensor], compute_ibp_from_affine: bool = True, tight: bool = True
    ) -> List[keras.KerasTensor]:
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
        self, inputsformode: List[keras.KerasTensor], compute_ibp_from_affine: bool = False, tight: bool = True
    ) -> List[List[keras.KerasTensor]]:
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

    def split_inputsformode_to_merge(self, inputsformode: List[Any]) -> List[List[Any]]:
        n_comp = self.nb_tensors
        return [inputsformode[n_comp * i : n_comp * (i + 1)] for i in range(len(inputsformode) // n_comp)]

    def extract_inputsformode_from_fullinputs(self, inputs: List[keras.KerasTensor]) -> List[keras.KerasTensor]:
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

    def extract_outputsformode_from_fulloutputs(self, outputs: List[keras.KerasTensor]) -> List[keras.KerasTensor]:
        return self.extract_inputsformode_from_fullinputs(outputs)

    @staticmethod
    def get_empty_tensor(dtype: Optional[str] = None) -> keras.KerasTensor:
        if dtype is None:
            dtype = floatx()
        return K.convert_to_tensor([], dtype=dtype)


def get_upper_box(
    x_min: keras.KerasTensor, x_max: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
    """#compute the max of an affine function
    within a box (hypercube) defined by its extremal corners

    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w: weights of the affine function
        b: bias of the affine function

    Returns:
        max_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):  # identity function
        return x_max

    # split into positive and negative components
    z_value = K.cast(0.0, dtype=x_min.dtype)
    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_out = x_min + z_value * x_min
    x_max_out = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_out = K.expand_dims(x_min_out, -1)
        x_max_out = K.expand_dims(x_max_out, -1)

    return K.sum(w_pos * x_max_out + w_neg * x_min_out, 1) + b


def get_lower_box(
    x_min: keras.KerasTensor, x_max: keras.KerasTensor, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
    """
    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w_l: weights of the affine lower bound
        b_l: bias of the affine lower bound

    Returns:
        min_(x >= x_min, x<=x_max) w*x + b
    """

    if len(w.shape) == len(b.shape):
        return x_min

    z_value = K.cast(0.0, dtype=x_min.dtype)

    w_pos = K.maximum(w, z_value)
    w_neg = K.minimum(w, z_value)

    x_min_out = x_min + z_value * x_min
    x_max_out = x_max + z_value * x_max

    for _ in range(len(w.shape) - len(x_max.shape)):
        x_min_out = K.expand_dims(x_min_out, -1)
        x_max_out = K.expand_dims(x_max_out, -1)

    return K.sum(w_pos * x_min_out + w_neg * x_max_out, 1) + b


def get_lq_norm(x: keras.KerasTensor, p: float, axis: int = -1) -> keras.KerasTensor:
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


def get_upper_ball(
    x_0: keras.KerasTensor, eps: float, p: float, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
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
        return x_0 + eps

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


def get_lower_ball(
    x_0: keras.KerasTensor, eps: float, p: float, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
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


def get_lower_ball_finetune(
    x_0: keras.KerasTensor, eps: float, p: float, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
    if "finetune_lower" in kwargs and "upper" in kwargs or "lower" in kwargs:
        alpha = kwargs["finetune_lower"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * lower_reshaped, 1) + K.sum(
                K.minimum(0.0, w_alpha_bar) * upper_reshaped, 1
            )
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(0, w) * alpha[None] + K.maximum(0.0, w)
            w_alpha_bar = K.minimum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(0.0, w_alpha_bar) * upper_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:
            lower = kwargs["lower"]  # flatten vector
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(0, w) * alpha[None] + K.minimum(0.0, w)
            w_alpha_bar = K.maximum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * lower_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_lower_ball(x_0, eps, p, w, b)


def get_upper_ball_finetune(
    x_0: keras.KerasTensor, eps: float, p: float, w: keras.KerasTensor, b: keras.KerasTensor, **kwargs: Any
) -> keras.KerasTensor:
    if "finetune_upper" in kwargs and "upper" in kwargs or "lower" in kwargs:
        alpha = kwargs["finetune_upper"]
        # assume alpha is the same shape as w, minus the batch dimension
        n_shape = len(w.shape) - 2

        if "upper" and "lower" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            lower = kwargs["lower"]  # flatten vector

            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = w * alpha[None]
            w_alpha_bar = w * (1 - alpha)

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * upper_reshaped, 1) + K.sum(
                K.minimum(0.0, w_alpha_bar) * lower_reshaped, 1
            )
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "upper" in kwargs:
            upper = kwargs["upper"]  # flatten vector
            upper_reshaped = np.reshape(upper, [1, -1] + [1] * n_shape)

            w_alpha = K.minimum(0, w) * alpha[None] + K.maximum(0.0, w)
            w_alpha_bar = K.minimum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.maximum(0.0, w_alpha_bar) * upper_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

        if "lower" in kwargs:
            lower = kwargs["lower"]  # flatten vector
            lower_reshaped = np.reshape(lower, [1, -1] + [1] * n_shape)

            w_alpha = K.maximum(0, w) * alpha[None] + K.minimum(0.0, w)
            w_alpha_bar = K.maximum(0, w) * (1 - alpha[None])

            score_box = K.sum(K.minimum(0.0, w_alpha_bar) * lower_reshaped, 1)
            score_ball = get_lower_ball(x_0, eps, p, w_alpha, b)

            return score_box + score_ball

    return get_upper_ball(x_0, eps, p, w, b)
