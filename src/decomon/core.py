from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


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

    def get_input_shape(self, inputsformode: List[tf.Tensor]) -> tf.TensorShape:
        return inputsformode[-1].shape

    def get_fullinputs_from_inputsformode(self, inputsformode: List[tf.Tensor]) -> List[tf.Tensor]:
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

        return [x, u_c, w_u, b_u, l_c, w_l, b_l, h, g]

    def extract_inputsformode_from_fullinputs(self, inputs: List[tf.Tensor]) -> List[tf.Tensor]:
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

    def extract_outputsformode_from_fulloutputs(self, outputs: List[tf.Tensor]) -> List[tf.Tensor]:
        return self.extract_inputsformode_from_fullinputs(outputs)

    @staticmethod
    def get_empty_tensor(dtype: Union[str, tf.DType] = K.floatx()) -> tf.Tensor:
        return tf.constant([], dtype=dtype)
