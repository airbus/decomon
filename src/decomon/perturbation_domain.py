from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union

import numpy as np
from keras import ops as K

from decomon.keras_utils import batch_multid_dot
from decomon.types import Tensor


class Option(str, Enum):
    lagrangian = "lagrangian"
    milp = "milp"


class PerturbationDomain(ABC):
    opt_option: Option

    def __init__(self, opt_option: Union[str, Option] = Option.milp):
        self.opt_option = Option(opt_option)

    @abstractmethod
    def get_upper_x(self, x: Tensor) -> Tensor:
        """Get upper constant bound on perturbation domain input."""
        ...

    @abstractmethod
    def get_lower_x(self, x: Tensor) -> Tensor:
        """Get lower constant bound on perturbation domain input."""
        ...

    @abstractmethod
    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        """Merge upper affine bounds with perturbation domain input to get upper constant bound.

        Args:
            x: perturbation domain input
            w: weights of the affine bound
            b: bias of the affine bound
            missing_batchsize: whether w and b are missing batchsize
            **kwargs:

        Returns:

        """
        ...

    @abstractmethod
    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        """Merge lower affine bounds with perturbation domain input to get lower constant bound.

        Args:
            x: perturbation domain input
            w: weights of the affine bound
            b: bias of the affine bound
            missing_batchsize: whether w and b are missing batchsize
            **kwargs:

        Returns:

        """
        ...

    @abstractmethod
    def get_nb_x_components(self) -> int:
        """Get the number of components in perturabation domain input.

        For instance:
        - box domain: each corner of the box -> 2 components
        - ball domain: center of the ball -> 1 component

        """
        ...

    def get_config(self) -> dict[str, Any]:
        return {
            "opt_option": self.opt_option,
        }

    def get_kerasinputlike_from_x(self, x: Tensor) -> Tensor:
        """Get tensor of same shape as keras model input, from perturbation domain input x

        Args:
            x: perturbation domain input

        Returns:
            tensor of same shape as keras model input

        """
        if self.get_nb_x_components() == 1:
            return x
        else:
            return x[:, 0]

    def get_x_input_shape_wo_batchsize(self, original_input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get expected perturbation domain input shape, excepting the batch axis."""
        n_comp_x = self.get_nb_x_components()
        if n_comp_x == 1:
            return original_input_shape
        else:
            return (n_comp_x,) + original_input_shape

    def get_keras_input_shape_wo_batchsize(self, x_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Deduce keras model input shape from perturbation domain input shape."""
        n_comp_x = self.get_nb_x_components()
        if n_comp_x == 1:
            return x_shape
        else:
            return x_shape[1:]


class BoxDomain(PerturbationDomain):
    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_upper_box(x_min=x_min, x_max=x_max, w=w, b=b, missing_batchsize=missing_batchsize, **kwargs)

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        x_min = x[:, 0]
        x_max = x[:, 1]
        return get_lower_box(x_min=x_min, x_max=x_max, w=w, b=b, missing_batchsize=missing_batchsize, **kwargs)

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

    def get_lower(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        return get_lower_ball(x_0=x, eps=self.eps, p=self.p, w=w, b=b, missing_batchsize=missing_batchsize, **kwargs)

    def get_upper(self, x: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
        return get_upper_ball(x_0=x, eps=self.eps, p=self.p, w=w, b=b, missing_batchsize=missing_batchsize, **kwargs)

    def get_nb_x_components(self) -> int:
        return 1

    def get_lower_x(self, x: Tensor) -> Tensor:
        return x - self.eps

    def get_upper_x(self, x: Tensor) -> Tensor:
        return x + self.eps


def get_upper_box(x_min: Tensor, x_max: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
    """Compute the max of an affine function
    within a box (hypercube) defined by its extremal corners

    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w: weights of the affine function
        b: bias of the affine function
        missing_batchsize: whether w and b are missing the batchsize

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
    diagonal = (False, is_diag)
    missing_batchsize = (False, missing_batchsize)

    return (
        batch_multid_dot(x_max, w_pos, diagonal=diagonal, missing_batchsize=missing_batchsize)
        + batch_multid_dot(x_min, w_neg, diagonal=diagonal, missing_batchsize=missing_batchsize)
        + b
    )


def get_lower_box(x_min: Tensor, x_max: Tensor, w: Tensor, b: Tensor, missing_batchsize=False, **kwargs: Any) -> Tensor:
    """
    Args:
        x_min: lower bound of the box domain
        x_max: upper bound of the box domain
        w: weights of the affine lower bound
        b: bias of the affine lower bound
        missing_batchsize: whether w and b are missing the batchsize

    Returns:
        min_(x >= x_min, x<=x_max) w*x + b

    Note:
        We can have w, b in diagonal representation and/or without a batch axis.
        We assume that x_min, x_max have always its batch axis.

    """
    return get_upper_box(x_min=x_max, x_max=x_min, w=w, b=b, missing_batchsize=missing_batchsize, **kwargs)


def get_lq_norm(x: Tensor, p: float, axis: Union[int, list[int]] = -1) -> Tensor:
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
    x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, missing_batchsize: bool = False, **kwargs: Any
) -> Tensor:
    """max of an affine function over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function
        missing_batchsize: whether w and b are missing the batchsize

    Returns:
        max_(|x - x_0|_p<= eps) w*x + b
    """
    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_upper_box(x_min, x_max, w, b, missing_batchsize=missing_batchsize)

    else:
        if len(kwargs):
            return get_upper_ball_finetune(x_0, eps, p, w, b, missing_batchsize=missing_batchsize, **kwargs)

        # use Holder's inequality p+q=1
        # ||w||_q*eps + w*x_0 + b

        is_diag = w.shape == b.shape

        # lq-norm of w
        if is_diag:
            w_q = K.abs(w)
        else:
            nb_axes_wo_batchsize_x = len(x_0.shape) - 1
            if missing_batchsize:
                reduced_axes = list(range(nb_axes_wo_batchsize_x))
            else:
                reduced_axes = list(range(1, 1 + nb_axes_wo_batchsize_x))
            w_q = get_lq_norm(w, p, axis=reduced_axes)

        diagonal = (False, is_diag)
        missing_batchsize = (False, missing_batchsize)
        return batch_multid_dot(x_0, w, diagonal=diagonal, missing_batchsize=missing_batchsize) + b + w_q * eps


def get_lower_ball(
    x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, missing_batchsize: bool = False, **kwargs: Any
) -> Tensor:
    """min of an affine fucntion over an Lp ball

    Args:
        x_0: the center of the ball
        eps: the radius
        p: the type of Lp norm considered
        w: weights of the affine function
        b: bias of the affine function
        missing_batchsize: whether w and b are missing the batchsize

    Returns:
        min_(|x - x_0|_p<= eps) w*x + b
    """
    if p == np.inf:
        # compute x_min and x_max according to eps
        x_min = x_0 - eps
        x_max = x_0 + eps
        return get_lower_box(x_min, x_max, w, b, missing_batchsize=missing_batchsize)

    else:
        if len(kwargs):
            return get_lower_ball_finetune(x_0, eps, p, w, b, missing_batchsize=missing_batchsize, **kwargs)

        # use Holder's inequality p+q=1
        # - ||w||_q*eps + w*x_0 + b

        is_diag = w.shape == b.shape

        # lq-norm of w
        if is_diag:
            w_q = K.abs(w)
        else:
            nb_axes_wo_batchsize_x = len(x_0.shape) - 1
            if missing_batchsize:
                reduced_axes = list(range(nb_axes_wo_batchsize_x))
            else:
                reduced_axes = list(range(1, 1 + nb_axes_wo_batchsize_x))
            w_q = get_lq_norm(w, p, axis=reduced_axes)

        diagonal = (False, is_diag)
        missing_batchsize = (False, missing_batchsize)
        return batch_multid_dot(x_0, w, diagonal=diagonal, missing_batchsize=missing_batchsize) + b - w_q * eps


def get_lower_ball_finetune(
    x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, missing_batchsize: bool = False, **kwargs: Any
) -> Tensor:
    if missing_batchsize:
        raise NotImplementedError()

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


def get_upper_ball_finetune(
    x_0: Tensor, eps: float, p: float, w: Tensor, b: Tensor, missing_batchsize: bool = False, **kwargs: Any
) -> Tensor:
    if missing_batchsize:
        raise NotImplementedError()

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
