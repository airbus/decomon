from enum import Enum
from typing import Union

import numpy as np


class Option(Enum):
    lagrangian = "lagrangian"
    milp = "milp"


class Slope(Enum):
    V_SLOPE = "volume-slope"
    A_SLOPE = "adaptative-slope"
    S_SLOPE = "same-slope"
    Z_SLOPE = "zero-lb"
    O_SLOPE = "one-lb"


class PerturbationDomain:
    opt_option: Option

    def __init__(self, opt_option: Union[str, Option] = Option.milp):
        self.opt_option = Option(opt_option)


class BoxDomain(PerturbationDomain):
    pass


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


class ForwardMode(Enum):
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


class InputsOutputsSpec:
    """Storing specifications for inputs and outputs of decomon/backward layer/model."""

    def __init__(self, dc_decomp: bool = False, mode: Union[str, ForwardMode] = ForwardMode.HYBRID):
        """
        Args:
            dc_decomp: boolean that indicates whether we return a
                difference of convex decomposition of our layer
            mode: type of Forward propagation (ibp, affine, or hybrid)
        gradient
        """

        self.mode = ForwardMode(mode)
        self.dc_decomp = dc_decomp

    @property
    def nb_tensors(self):
        if self.mode == ForwardMode.HYBRID:
            nb_tensors = 7
        elif self.mode == ForwardMode.IBP:
            nb_tensors = 2
        elif self.mode == ForwardMode.AFFINE:
            nb_tensors = 5
        else:
            raise NotImplementedError(f"unknown forward mode {mode}")

        if self.dc_decomp:
            nb_tensors += 2

        return nb_tensors
