from typing import List, Optional, Union

import keras_core as keras
import keras_core.ops as K
import numpy as np
import numpy.typing as npt
import pytest
from keras_core import Input
from keras_core.layers import (
    Activation,
    Add,
    Average,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Reshape,
)
from keras_core.models import Model, Sequential
from numpy.testing import assert_almost_equal

from decomon.core import ForwardMode, Slope
from decomon.models.utils import ConvertMethod


@pytest.fixture(params=[m.value for m in ForwardMode])
def mode(request):
    return request.param


@pytest.fixture(params=[s.value for s in Slope])
def slope(request):
    return request.param


@pytest.fixture(params=list(range(10)))
def n(request):
    return request.param


@pytest.fixture(params=[32, 64, 16])
def floatx(request):
    # setup
    eps_bak = K.epsilon()  # store current epsilon
    floatx_bak = K.floatx()  # store current floatx
    precision = request.param
    K.set_floatx("float{}".format(precision))
    if precision == 16:
        K.set_epsilon(1e-2)
    # actual value
    yield precision
    # tear down
    K.set_epsilon(eps_bak)
    K.set_floatx(floatx_bak)


@pytest.fixture
def decimal(floatx):
    if floatx == 16:
        return 2
    else:
        return 4


@pytest.fixture(params=[True, False])
def dc_decomp(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_bias(request):
    return request.param


@pytest.fixture(params=[True, False])
def shared(request):
    return request.param


@pytest.fixture(params=["same", "valid"])
def padding(request):
    return request.param


@pytest.fixture(params=[None, "linear", "relu"])
def activation(request):
    return request.param


@pytest.fixture(params=["channels_last", "channels_first"])
def data_format(request):
    return request.param


@pytest.fixture(params=[0, 1])
def odd(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, -1])
def axis(request):
    return request.param


@pytest.fixture(params=[1, 2, 11])
def channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def filter_size(request):
    return request.param


@pytest.fixture(params=[1, 2])
def strides(request):
    return request.param


@pytest.fixture(params=[True, False])
def flatten(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3, 4, 5, 6])
def finetune_odd(request) -> Optional[np.ndarray]:
    # hard code several configuration of finetune for images odd=1 (6, 6, 2)
    finetune_params = np.zeros((6, 6, 2))
    if request.param == 1:
        finetune_params += 1
    elif request.param == 2:
        finetune_params[0] = 1
    elif request.param == 3:
        finetune_params[:, 0] = 1
    elif request.param == 4:
        finetune_params[:, :, 0] = 1
    elif request.param == 5:
        finetune_params[0, 0, 0] = 1
    else:
        return None

    return finetune_params


@pytest.fixture(params=[m.value for m in ConvertMethod])
def method(request):
    return request.param


class Helpers:
    @staticmethod
    def is_method_mode_compatible(method, mode):
        return not (
            ConvertMethod(method) in {ConvertMethod.CROWN_FORWARD_IBP, ConvertMethod.FORWARD_IBP}
            and ForwardMode(mode) != ForwardMode.IBP
        )

    @staticmethod
    def get_standard_values_1d_box(n, dc_decomp=True, grad_bounds=False, nb=100):
        """A set of functions with their monotonic decomposition for testing the activations"""
        w_u_ = np.ones(nb, dtype=K.floatx())
        b_u_ = np.zeros(nb, dtype=K.floatx())
        w_l_ = np.ones(nb, dtype=K.floatx())
        b_l_ = np.zeros(nb, dtype=K.floatx())

        if n == 0:
            # identity
            y_ = np.linspace(-2, -1, nb)
            x_ = np.linspace(-2, -1, nb)
            h_ = np.linspace(-2, -1, nb)
            g_ = np.zeros_like(x_)

        elif n == 1:
            y_ = np.linspace(1, 2, nb)
            x_ = np.linspace(1, 2, nb)
            h_ = np.linspace(1, 2, nb)
            g_ = np.zeros_like(x_)

        elif n == 2:
            y_ = np.linspace(-1, 1, nb)
            x_ = np.linspace(-1, 1, nb)
            h_ = np.linspace(-1, 1, nb)
            g_ = np.zeros_like(x_)

        elif n == 3:
            # identity
            y_ = np.linspace(-2, -1, nb)
            x_ = np.linspace(-2, -1, nb)
            h_ = 2 * np.linspace(-2, -1, nb)
            g_ = -np.linspace(-2, -1, nb)

        elif n == 4:
            y_ = np.linspace(1, 2, nb)
            x_ = np.linspace(1, 2, nb)
            h_ = 2 * np.linspace(1, 2, nb)
            g_ = -np.linspace(1, 2, nb)

        elif n == 5:
            y_ = np.linspace(-1, 1, nb)
            x_ = np.linspace(-1, 1, nb)
            h_ = 2 * np.linspace(-1, 1, nb)
            g_ = -np.linspace(-1, 1, nb)

        elif n == 6:
            assert nb == 100, "expected nb=100 samples"
            # cosine function
            x_ = np.linspace(-np.pi, np.pi, 100)
            y_ = np.cos(x_)
            h_ = np.concatenate([y_[:50], np.ones((50,))]) - 0.5
            g_ = np.concatenate([np.ones((50,)), y_[50:]]) - 0.5
            w_u_ = np.zeros_like(x_)
            w_l_ = np.zeros_like(x_)
            b_u_ = np.ones_like(x_)
            b_l_ = -np.ones_like(x_)

        elif n == 7:
            # h and g >0
            h_ = np.linspace(0.5, 2, nb)
            g_ = np.linspace(1, 2, nb)[::-1]
            x_ = h_ + g_
            y_ = h_ + g_

        elif n == 8:
            # h <0 and g <0
            # h_max+g_max <=0
            h_ = np.linspace(-2, -1, nb)
            g_ = np.linspace(-2, -1, nb)[::-1]
            y_ = h_ + g_
            x_ = h_ + g_

        elif n == 9:
            # h >0 and g <0
            # h_min+g_min >=0
            h_ = np.linspace(4, 5, nb)
            g_ = np.linspace(-2, -1, nb)[::-1]
            y_ = h_ + g_
            x_ = h_ + g_

        else:
            raise ValueError("n must be between 0 and 9.")

        x_min_ = x_.min() + np.zeros_like(x_)
        x_max_ = x_.max() + np.zeros_like(x_)

        x_0_ = np.concatenate([x_min_[:, None], x_max_[:, None]], 1)

        u_c_ = np.max(y_) * np.ones((nb,))
        l_c_ = np.min(y_) * np.ones((nb,))

        if dc_decomp:
            output = [
                x_[:, None],
                y_[:, None],
                x_0_[:, :, None],
                u_c_[:, None],
                w_u_[:, None, None],
                b_u_[:, None],
                l_c_[:, None],
                w_l_[:, None, None],
                b_l_[:, None],
                h_[:, None],
                g_[:, None],
            ]
        else:
            output = [
                x_[:, None],
                y_[:, None],
                x_0_[:, :, None],
                u_c_[:, None],
                w_u_[:, None, None],
                b_u_[:, None],
                l_c_[:, None],
                w_l_[:, None, None],
                b_l_[:, None],
            ]

        # cast element
        return [e.astype(K.floatx()) for e in output]

    @staticmethod
    def get_inputs_for_mode_from_full_inputs(
        inputs: Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]],
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        dc_decomp: bool = True,
    ) -> Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]:
        """Extract from full inputs the ones corresponding to the selected mode.

        Args:
            inputs: inputs from `get_standard_values_xxx()` or `get_tensor_decomposition_xxx()`
            mode:
            dc_decomp:

        Returns:

        """
        mode = ForwardMode(mode)
        if dc_decomp:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
            if mode == ForwardMode.HYBRID:
                return inputs[2:]
            elif mode == ForwardMode.AFFINE:
                return [z, W_u, b_u, W_l, b_l, h, g]
            elif mode == ForwardMode.IBP:
                return [u_c, l_c, h, g]
            else:
                raise ValueError("Unknown mode.")
        else:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
            if mode == ForwardMode.HYBRID:
                return inputs[2:]
            elif mode == ForwardMode.AFFINE:
                return [z, W_u, b_u, W_l, b_l]
            elif mode == ForwardMode.IBP:
                return [u_c, l_c]
            else:
                raise ValueError("Unknown mode.")

    @staticmethod
    def get_inputs_np_for_decomon_model_from_full_inputs(
        inputs: List[npt.NDArray[np.float_]],
    ) -> npt.NDArray[np.float_]:
        """Extract from full numpy inputs the ones for a decomon model prediction.

        Args:
            inputs:  inputs from `get_standard_values_xxx()`

        Returns:

        """
        l_c_, u_c_ = Helpers.get_input_ref_bounds_from_full_inputs(inputs=inputs)
        return np.concatenate((l_c_[:, None], u_c_[:, None]), axis=1)

    @staticmethod
    def get_input_ref_bounds_from_full_inputs(
        inputs: Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]],
    ) -> Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]:
        """Extract lower and upper bound for input ref from full inputs

        Args:
            inputs: inputs from `get_standard_values_xxx()` or `get_tensor_decomposition_xxx()`

        Returns:

        """
        u_c_, l_c_ = inputs[3], inputs[6]
        return [l_c_, u_c_]

    @staticmethod
    def prepare_full_np_inputs_for_convert_model(
        inputs: List[npt.NDArray[np.float_]],
        dc_decomp: bool = True,
    ) -> List[npt.NDArray[np.float_]]:
        """Prepare full numpy inputs for convert_forward or convert_backward.

        W_u and W_l will be idendity matrices, and b_u, b_l zeros vectors.

        Args:
            inputs: inputs from `get_standard_values_xxx()` or `get_tensor_decomposition_xxx()`

        Returns:

        """
        if dc_decomp:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
        else:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs

        b_u = np.zeros_like(b_u)
        b_l = np.zeros_like(b_l)
        W_u = np.repeat(np.identity(n=W_u.shape[-1])[None, :, :], repeats=W_u.shape[0], axis=0)
        W_l = np.repeat(np.identity(n=W_l.shape[-1])[None, :, :], repeats=W_l.shape[0], axis=0)

        if dc_decomp:
            return [x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g]
        else:
            return [x, y, z, u_c, W_u, b_u, l_c, W_l, b_l]

    @staticmethod
    def get_input_tensors_for_decomon_convert_from_full_inputs(
        inputs: List[keras.KerasTensor],
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        dc_decomp: bool = True,
    ) -> List[keras.KerasTensor]:
        """Extract from full tensor inputs the ones for a conversion to decomon model.

        Args:
            inputs:  inputs from `get_tensor_decomposition_xxx()`
            mode:
            dc_decomp:

        Returns:

        """
        mode = ForwardMode(mode)
        if dc_decomp:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l, h, g = inputs
            input_box_tensor = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            if mode == ForwardMode.HYBRID:
                return [input_box_tensor, u_c, W_u, b_u, l_c, W_l, b_l, h, g]
            elif mode == ForwardMode.AFFINE:
                return [input_box_tensor, W_u, b_u, W_l, b_l, h, g]
            elif mode == ForwardMode.IBP:
                return [u_c, l_c, h, g]
            else:
                raise ValueError("Unknown mode.")
        else:
            x, y, z, u_c, W_u, b_u, l_c, W_l, b_l = inputs
            input_box_tensor = K.concatenate([K.expand_dims(l_c, 1), K.expand_dims(u_c, 1)], 1)
            if mode == ForwardMode.HYBRID:
                return [input_box_tensor, u_c, W_u, b_u, l_c, W_l, b_l]
            elif mode == ForwardMode.AFFINE:
                return [input_box_tensor, W_u, b_u, W_l, b_l]
            elif mode == ForwardMode.IBP:
                return [u_c, l_c]
            else:
                raise ValueError("Unknown mode.")

    @staticmethod
    def get_input_ref_from_full_inputs(
        inputs: Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]
    ) -> Union[keras.KerasTensor, npt.NDArray[np.float_]]:
        """Extract from full inputs the input of reference for the original Keras layer.

        Args:
            inputs: inputs from `get_standard_values_xxx()` or `get_tensor_decomposition_xxx()`

        Returns:

        """
        return inputs[1]

    @staticmethod
    def get_tensor_decomposition_1d_box(dc_decomp=True):
        if dc_decomp:
            return [
                Input((1,), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((2, 1), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((1, 1), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((1, 1), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
                Input((1,), dtype=K.floatx()),
            ]
        return [
            Input((1,), dtype=K.floatx()),
            Input((1,), dtype=K.floatx()),
            Input((2, 1), dtype=K.floatx()),
            Input((1,), dtype=K.floatx()),
            Input((1, 1), dtype=K.floatx()),
            Input((1,), dtype=K.floatx()),
            Input((1,), dtype=K.floatx()),
            Input((1, 1), dtype=K.floatx()),
            Input((1,), dtype=K.floatx()),
        ]

    @staticmethod
    def get_full_outputs_from_outputs_for_mode(
        outputs_for_mode: Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]],
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        dc_decomp: bool = True,
        full_inputs: Optional[Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]] = None,
    ) -> Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]:
        mode = ForwardMode(mode)
        if dc_decomp:
            if mode == ForwardMode.HYBRID:
                z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = outputs_for_mode
            elif mode == ForwardMode.AFFINE:
                z_, w_u_, b_u_, w_l_, b_l_, h_, g_ = outputs_for_mode
                u_c_, l_c_ = None, None
            elif mode == ForwardMode.IBP:
                u_c_, l_c_, h_, g_ = outputs_for_mode
                z_, w_u_, b_u_, w_l_, b_l_ = None, None, None, None, None
                if full_inputs is not None:
                    z_ = full_inputs[2]
            else:
                raise ValueError("Unknown mode.")
        else:
            if mode == ForwardMode.HYBRID:
                z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = outputs_for_mode
            elif mode == ForwardMode.AFFINE:
                z_, w_u_, b_u_, w_l_, b_l_ = outputs_for_mode
                u_c_, l_c_ = None, None
            elif mode == ForwardMode.IBP:
                u_c_, l_c_ = outputs_for_mode
                z_, w_u_, b_u_, w_l_, b_l_ = None, None, None, None, None
                if full_inputs is not None:
                    z_ = full_inputs[2]
            else:
                raise ValueError("Unknown mode.")
            h_, g_ = None, None
        return [z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]

    @staticmethod
    def get_input_dim_multid_box(odd):
        if odd:
            return 3
        else:
            return 2

    @staticmethod
    def get_input_dim_images_box(odd):
        if odd:
            return 7
        else:
            return 6

    @staticmethod
    def get_input_dim_from_full_inputs(inputs: Union[List[keras.KerasTensor], List[npt.NDArray[np.float_]]]) -> int:
        """Get input_dim for to_decomon or to_backward from full inputs

        Args:
            inputs: inputs from `get_standard_values_xxx()` or `get_tensor_decomposition_xxx()`

        Returns:

        """
        return inputs[0].shape[-1]

    @staticmethod
    def get_tensor_decomposition_multid_box(odd=1, dc_decomp=True):
        n = Helpers.get_input_dim_multid_box(odd)

        if dc_decomp:
            # x, y, z, u, w_u, b_u, l, w_l, b_l, h, g
            return [
                Input((n,), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((2, n), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((n, n), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((n, n), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
                Input((n,), dtype=K.floatx()),
            ]
        return [
            Input((n,), dtype=K.floatx()),
            Input((n,), dtype=K.floatx()),
            Input((2, n), dtype=K.floatx()),
            Input((n,), dtype=K.floatx()),
            Input((n, n), dtype=K.floatx()),
            Input((n,), dtype=K.floatx()),
            Input((n,), dtype=K.floatx()),
            Input((n, n), dtype=K.floatx()),
            Input((n,), dtype=K.floatx()),
        ]

    @staticmethod
    def get_standard_values_multid_box(odd=1, dc_decomp=True):
        if dc_decomp:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
                h_0,
                g_0,
            ) = Helpers.get_standard_values_1d_box(0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
                h_1,
                g_1,
            ) = Helpers.get_standard_values_1d_box(1, dc_decomp)
        else:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
            ) = Helpers.get_standard_values_1d_box(0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
            ) = Helpers.get_standard_values_1d_box(1, dc_decomp)

        if not odd:
            # output (x_0+x_1, x_0+2*x_0)
            x_ = np.concatenate([x_0, x_1], -1)
            z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
            z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
            z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
            y_ = np.concatenate([y_0 + y_1, y_0 + 2 * y_1], -1)
            b_u_ = np.concatenate([b_u_0 + b_u_1, b_u_0 + 2 * b_u_1], -1)
            u_c_ = np.concatenate([u_c_0 + u_c_1, u_c_0 + 2 * u_c_1], -1)
            b_l_ = np.concatenate([b_l_0 + b_l_1, b_l_0 + 2 * b_l_1], -1)
            l_c_ = np.concatenate([l_c_0 + l_c_1, l_c_0 + 2 * l_c_1], -1)

            if dc_decomp:
                h_ = np.concatenate([h_0 + h_1, h_0 + 2 * h_1], -1)
                g_ = np.concatenate([g_0 + g_1, g_0 + 2 * g_1], -1)

            w_u_ = np.zeros((len(x_), 2, 2))
            w_u_[:, 0, 0] = w_u_0[:, 0, 0]
            w_u_[:, 1, 0] = w_u_1[:, 0, 0]
            w_u_[:, 0, 1] = w_u_0[:, 0, 0]
            w_u_[:, 1, 1] = 2 * w_u_1[:, 0, 0]

            w_l_ = np.zeros((len(x_), 2, 2))
            w_l_[:, 0, 0] = w_l_0[:, 0, 0]
            w_l_[:, 1, 0] = w_l_1[:, 0, 0]
            w_l_[:, 0, 1] = w_l_0[:, 0, 0]
            w_l_[:, 1, 1] = 2 * w_l_1[:, 0, 0]

        else:
            (
                x_2,
                y_2,
                z_2,
                u_c_2,
                w_u_2,
                b_u_2,
                l_c_2,
                w_l_2,
                b_l_2,
                h_2,
                g_2,
            ) = Helpers.get_standard_values_1d_box(2)

            # output (x_0+x_1, x_0+2*x_0, x_2)
            x_ = np.concatenate([x_0, x_1, x_2], -1)
            z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0], z_2[:, 0]], -1)
            z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1], z_2[:, 1]], -1)
            z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
            y_ = np.concatenate([y_0 + y_1, y_0 + 2 * y_1, y_2], -1)
            b_u_ = np.concatenate([b_u_0 + b_u_1, b_u_0 + 2 * b_u_1, b_u_2], -1)
            b_l_ = np.concatenate([b_l_0 + b_l_1, b_l_0 + 2 * b_l_1, b_l_2], -1)
            u_c_ = np.concatenate([u_c_0 + u_c_1, u_c_0 + 2 * u_c_1, u_c_2], -1)
            l_c_ = np.concatenate([l_c_0 + l_c_1, l_c_0 + 2 * l_c_1, l_c_2], -1)

            if dc_decomp:
                h_ = np.concatenate([h_0 + h_1, h_0 + 2 * h_1, h_2], -1)
                g_ = np.concatenate([g_0 + g_1, g_0 + 2 * g_1, g_2], -1)

            w_u_ = np.zeros((len(x_), 3, 3))
            w_u_[:, 0, 0] = w_u_0[:, 0, 0]
            w_u_[:, 1, 0] = w_u_1[:, 0, 0]
            w_u_[:, 0, 1] = w_u_0[:, 0, 0]
            w_u_[:, 1, 1] = 2 * w_u_1[:, 0, 0]
            w_u_[:, 2, 2] = w_u_2[:, 0, 0]

            w_l_ = np.zeros((len(x_), 3, 3))
            w_l_[:, 0, 0] = w_l_0[:, 0, 0]
            w_l_[:, 1, 0] = w_l_1[:, 0, 0]
            w_l_[:, 0, 1] = w_l_0[:, 0, 0]
            w_l_[:, 1, 1] = 2 * w_l_1[:, 0, 0]
            w_l_[:, 2, 2] = w_l_2[:, 0, 0]

        if dc_decomp:
            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def build_image_from_1D_box(odd=0, m=0, dc_decomp=True):
        n = Helpers.get_input_dim_images_box(odd)

        if dc_decomp:
            (
                x_,
                y_0,
                z_,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
                h_0,
                g_0,
            ) = Helpers.get_standard_values_1d_box(m, dc_decomp=dc_decomp)
        else:
            (
                x_,
                y_0,
                z_,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
            ) = Helpers.get_standard_values_1d_box(m, dc_decomp=dc_decomp)

        y_ = np.concatenate([(i + 1) * y_0 for i in range(n * n)], -1).reshape((-1, n, n))
        b_u_ = np.concatenate([(i + 1) * b_u_0 for i in range(n * n)], -1).reshape((-1, n, n))
        b_l_ = np.concatenate([(i + 1) * b_l_0 for i in range(n * n)], -1).reshape((-1, n, n))

        if dc_decomp:
            h_ = np.concatenate([(i + 1) * h_0 for i in range(n * n)], -1).reshape((-1, n, n))
            g_ = np.concatenate([(i + 1) * g_0 for i in range(n * n)], -1).reshape((-1, n, n))

        u_c_ = np.concatenate([(i + 1) * u_c_0 for i in range(n * n)], -1).reshape((-1, n, n))
        l_c_ = np.concatenate([(i + 1) * l_c_0 for i in range(n * n)], -1).reshape((-1, n, n))

        w_u_ = np.zeros((len(x_), 1, n * n))
        w_l_ = np.zeros((len(x_), 1, n * n))

        for i in range(n * n):
            w_u_[:, 0, i] = (i + 1) * w_u_0[:, 0, 0]
            w_l_[:, 0, i] = (i + 1) * w_l_0[:, 0, 0]

        w_u_ = w_u_.reshape((-1, 1, n, n))
        w_l_ = w_l_.reshape((-1, 1, n, n))

        if dc_decomp:
            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def build_image_from_2D_box(odd=0, m0=0, m1=1, dc_decomp=True):
        if dc_decomp:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
                h_0,
                g_0,
            ) = Helpers.build_image_from_1D_box(odd, m0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
                h_1,
                g_1,
            ) = Helpers.build_image_from_1D_box(odd, m1, dc_decomp)
        else:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
            ) = Helpers.build_image_from_1D_box(odd, m0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
            ) = Helpers.build_image_from_1D_box(odd, m1, dc_decomp)

        x_ = np.concatenate([x_0, x_1], -1)
        z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
        z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
        z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
        y_ = y_0 + y_1
        b_u_ = b_u_0 + b_u_1
        b_l_ = b_l_0 + b_l_1

        u_c_ = u_c_0 + u_c_1
        l_c_ = l_c_0 + l_c_1

        w_u_ = np.concatenate([w_u_0, w_u_1], 1)
        w_l_ = np.concatenate([w_l_0, w_l_1], 1)

        if dc_decomp:
            h_ = h_0 + h_1
            g_ = g_0 + g_1

            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def get_standard_values_images_box(data_format="channels_last", odd=0, m0=0, m1=1, dc_decomp=True):
        output = Helpers.build_image_from_2D_box(odd, m0, m1, dc_decomp)
        if dc_decomp:
            x_0, y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0, h_0, g_0 = output
        else:
            x_0, y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = output

        x_ = x_0
        z_ = z_0
        z_min_ = z_0[:, 0]
        z_max_ = z_0[:, 1]

        if data_format == "channels_last":
            y_0 = y_0[:, :, :, None]
            b_u_0 = b_u_0[:, :, :, None]
            b_l_0 = b_l_0[:, :, :, None]
            u_c_0 = u_c_0[:, :, :, None]
            l_c_0 = l_c_0[:, :, :, None]
            w_u_0 = w_u_0[:, :, :, :, None]
            w_l_0 = w_l_0[:, :, :, :, None]
            y_ = np.concatenate([y_0, y_0], -1)
            b_u_ = np.concatenate([b_u_0, b_u_0], -1)
            b_l_ = np.concatenate([b_l_0, b_l_0], -1)
            u_c_ = np.concatenate([u_c_0, u_c_0], -1)
            l_c_ = np.concatenate([l_c_0, l_c_0], -1)
            w_u_ = np.concatenate([w_u_0, w_u_0], -1)
            w_l_ = np.concatenate([w_l_0, w_l_0], -1)

            if dc_decomp:
                h_0 = h_0[:, :, :, None]
                g_0 = g_0[:, :, :, None]
                h_ = np.concatenate([h_0, h_0], -1)
                g_ = np.concatenate([g_0, g_0], -1)

        else:
            output = Helpers.get_standard_values_images_box(
                data_format="channels_last", odd=odd, m0=m0, m1=m1, dc_decomp=dc_decomp
            )

            x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output[:9]
            if dc_decomp:
                h_, g_ = output[-2:]
                h_ = np.transpose(h_, (0, 3, 1, 2))
                g_ = np.transpose(g_, (0, 3, 1, 2))
            y_ = np.transpose(y_, (0, 3, 1, 2))
            u_c_ = np.transpose(u_c_, (0, 3, 1, 2))
            l_c_ = np.transpose(l_c_, (0, 3, 1, 2))
            b_u_ = np.transpose(b_u_, (0, 3, 1, 2))
            b_l_ = np.transpose(b_l_, (0, 3, 1, 2))
            w_u_ = np.transpose(w_u_, (0, 1, 4, 2, 3))
            w_l_ = np.transpose(w_l_, (0, 1, 4, 2, 3))

        if dc_decomp:
            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
        else:
            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def get_tensor_decomposition_images_box(data_format, odd, dc_decomp=True):
        n = Helpers.get_input_dim_images_box(odd)

        if data_format == "channels_last":
            # x, y, z, u, w_u, b_u, l, w_l, b_l

            output = [
                Input((2,), dtype=K.floatx()),
                Input((n, n, 2), dtype=K.floatx()),
                Input((2, 2), dtype=K.floatx()),
                Input((n, n, 2), dtype=K.floatx()),
                Input((2, n, n, 2), dtype=K.floatx()),
                Input((n, n, 2), dtype=K.floatx()),
                Input((n, n, 2), dtype=K.floatx()),
                Input((2, n, n, 2), dtype=K.floatx()),
                Input((n, n, 2), dtype=K.floatx()),
            ]

            if dc_decomp:
                output += [Input((n, n, 2), dtype=K.floatx()), Input((n, n, 2), dtype=K.floatx())]
        else:
            output = [
                Input((2,), dtype=K.floatx()),
                Input((2, n, n), dtype=K.floatx()),
                Input((2, 2), dtype=K.floatx()),
                Input((2, n, n), dtype=K.floatx()),
                Input((2, 2, n, n), dtype=K.floatx()),
                Input((2, n, n), dtype=K.floatx()),
                Input((2, n, n), dtype=K.floatx()),
                Input((2, 2, n, n), dtype=K.floatx()),
                Input((2, n, n), dtype=K.floatx()),
            ]
            if dc_decomp:
                output += [Input((n, n, 2), dtype=K.floatx()), Input((n, n, 2), dtype=K.floatx())]

        return output

    @staticmethod
    def assert_output_properties_box(x_, y_, h_, g_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=4):
        if y_ is None:
            y_ = h_ + g_
        if h_ is not None:
            assert_almost_equal(h_ + g_, y_, decimal=decimal, err_msg="decomposition error")

        assert np.min(x_min_ <= x_max_), "x_min >x_max"

        assert_almost_equal(np.clip(x_min_ - x_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_min >x_")
        assert_almost_equal(np.clip(x_ - x_max_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_max < x_")
        if w_u_ is not None or w_l_ is not None:
            x_expand = x_ + np.zeros_like(x_)
            n_expand = len(w_u_.shape) - len(x_expand.shape)
            for i in range(n_expand):
                x_expand = np.expand_dims(x_expand, -1)

            if w_l_ is not None:
                lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
            if w_u_ is not None:
                upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

        # check that the functions h_ and g_ remains monotonic
        if h_ is not None:
            assert_almost_equal(
                np.clip(h_[:-1] - h_[1:], 0, np.inf),
                np.zeros_like(h_[1:]),
                decimal=decimal,
                err_msg="h is not increasing",
            )
            assert_almost_equal(
                np.clip(g_[1:] - g_[:-1], 0, np.inf),
                np.zeros_like(g_[1:]),
                decimal=decimal,
                err_msg="g is not increasing",
            )

        #
        if w_u_ is not None:
            if K.floatx() == "float32":
                assert_almost_equal(
                    np.clip(y_ - upper_, 0.0, 1e6),
                    np.zeros_like(y_),
                    decimal=decimal,
                    err_msg="upper <y",
                )
        if w_l_ is not None:
            if K.floatx() == "float32":
                assert_almost_equal(
                    np.clip(lower_ - y_, 0.0, np.inf),
                    np.zeros_like(y_),
                    decimal=decimal,
                    err_msg="lower_ >y",
                )

        if l_c_ is not None:
            assert_almost_equal(
                np.clip(l_c_ - y_, 0.0, np.inf),
                np.zeros_like(y_),
                decimal=decimal,
                err_msg="l_c >y",
            )
            assert_almost_equal(
                np.clip(y_ - u_c_, 0.0, 1e6),
                np.zeros_like(y_),
                decimal=decimal,
                err_msg="u_c <y",
            )

    @staticmethod
    def assert_decomon_model_output_properties_box(
        full_inputs, output_ref, outputs_for_mode, mode=ForwardMode.HYBRID, dc_decomp=True, decimal=4
    ):
        input_ref_ = Helpers.get_input_ref_from_full_inputs(full_inputs)
        input_ref_min_, input_ref_max_ = Helpers.get_input_ref_bounds_from_full_inputs(inputs=full_inputs)
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = Helpers.get_full_outputs_from_outputs_for_mode(
            outputs_for_mode=outputs_for_mode, mode=mode, dc_decomp=dc_decomp, full_inputs=full_inputs
        )

        Helpers.assert_output_properties_box(
            input_ref_,
            output_ref,
            h_,
            g_,
            input_ref_min_,
            input_ref_max_,
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            decimal=decimal,
        )

    @staticmethod
    def assert_decomon_layer_output_properties_box(
        full_inputs, output_ref, outputs_for_mode, mode=ForwardMode.HYBRID, dc_decomp=True, decimal=4
    ):
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = Helpers.get_full_outputs_from_outputs_for_mode(
            outputs_for_mode=outputs_for_mode, mode=mode, dc_decomp=dc_decomp, full_inputs=full_inputs
        )
        x_ = full_inputs[0]
        Helpers.assert_output_properties_box(
            x_,
            output_ref,
            h_,
            g_,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            decimal=decimal,
        )

    @staticmethod
    def assert_decomon_layer_output_properties_box_linear(
        full_inputs, output_ref, outputs_for_mode, mode=ForwardMode.HYBRID, dc_decomp=True, decimal=4
    ):
        z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_ = Helpers.get_full_outputs_from_outputs_for_mode(
            outputs_for_mode=outputs_for_mode, mode=mode, dc_decomp=dc_decomp, full_inputs=full_inputs
        )
        x_ = full_inputs[0]
        Helpers.assert_output_properties_box_linear(
            x_,
            output_ref,
            z_[:, 0],
            z_[:, 1],
            u_c_,
            w_u_,
            b_u_,
            l_c_,
            w_l_,
            b_l_,
            decimal=decimal,
        )

    @staticmethod
    def assert_backward_layer_output_properties_box_linear(
        full_inputs, backward_outputs, output_ref=None, upper_constant_bound=None, lower_constant_bound=None, decimal=4
    ):
        w_u_, b_u_, w_l_, b_l_ = backward_outputs
        x_, y_, z_, u_c_, W_u_, B_u_, l_c_, W_l_, B_l_ = full_inputs

        # backward recomposition
        w_u_b = np.sum(np.maximum(w_u_, 0) * W_u_ + np.minimum(w_u_, 0) * W_l_, 1)[:, :, None]
        b_u_b = (
            b_u_ + np.sum(np.maximum(w_u_, 0) * B_u_[:, :, None], 1) + np.sum(np.minimum(w_u_, 0) * B_l_[:, :, None], 1)
        )
        w_l_b = np.sum(np.maximum(w_l_, 0) * W_l_ + np.minimum(w_l_, 0) * W_u_, 1)[:, :, None]
        b_l_b = (
            b_l_ + np.sum(np.maximum(w_l_, 0) * B_l_[:, :, None], 1) + np.sum(np.minimum(w_l_, 0) * B_u_[:, :, None], 1)
        )

        Helpers.assert_output_properties_box_linear(
            x_,
            output_ref,
            z_[:, 0],
            z_[:, 1],
            upper_constant_bound,
            w_u_b,
            b_u_b,
            lower_constant_bound,
            w_l_b,
            b_l_b,
            decimal=decimal,
        )

    @staticmethod
    def assert_output_properties_box_linear(x_, y_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=4):
        # flatten everyting
        n = len(x_)
        if y_ is not None:
            n = len(y_)
            y_ = y_.reshape((n, -1))
        if l_c_ is not None:
            u_c_ = u_c_.reshape((n, -1))
            l_c_ = l_c_.reshape((n, -1))
        if w_u_ is not None:
            w_u_ = w_u_.reshape((n, w_u_.shape[1], -1))
            w_l_ = w_l_.reshape((n, w_l_.shape[1], -1))
            b_u_ = b_u_.reshape((n, -1))
            b_l_ = b_l_.reshape((n, -1))

        assert np.min(x_min_ <= x_max_), "x_min >x_max"

        assert_almost_equal(np.clip(x_min_ - x_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_min >x_")
        assert_almost_equal(np.clip(x_ - x_max_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_max < x_")
        if w_u_ is not None:
            x_expand = x_ + np.zeros_like(x_)
            n_expand = len(w_u_.shape) - len(x_expand.shape)
            for i in range(n_expand):
                x_expand = np.expand_dims(x_expand, -1)

            lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
            upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

        if y_ is not None:
            if l_c_ is not None:
                assert_almost_equal(
                    np.clip(l_c_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="l_c >y"
                )
                assert_almost_equal(np.clip(y_ - u_c_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="u_c <y")
            if w_u_ is not None:
                assert_almost_equal(
                    np.clip(lower_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="lower_ >y"
                )
                assert_almost_equal(
                    np.clip(y_ - upper_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="upper <y"
                )

        # computer lower bounds on the domain
        if w_u_ is not None:
            x_expand_min = x_min_ + np.zeros_like(x_)
            x_expand_max = x_max_ + np.zeros_like(x_)
            n_expand = len(w_u_.shape) - len(x_expand_min.shape)
            for i in range(n_expand):
                x_expand_min = np.expand_dims(x_expand_min, -1)
                x_expand_max = np.expand_dims(x_expand_max, -1)

            lower_ = (
                np.sum(np.maximum(0, w_l_) * x_expand_min, 1) + np.sum(np.minimum(0, w_l_) * x_expand_max, 1) + b_l_
            )
            upper_ = (
                np.sum(np.maximum(0, w_u_) * x_expand_max, 1) + np.sum(np.minimum(0, w_u_) * x_expand_min, 1) + b_u_
            )

            if y_ is not None:
                assert_almost_equal(
                    np.clip(lower_ - y_, 0.0, np.inf), np.zeros_like(y_), decimal=decimal, err_msg="l_c >y"
                )
                assert_almost_equal(
                    np.clip(y_ - upper_, 0.0, 1e6), np.zeros_like(y_), decimal=decimal, err_msg="u_c <y"
                )

    # multi decomposition for convert
    @staticmethod
    def get_standard_values_multid_box_convert(odd=1, dc_decomp=True):
        if dc_decomp:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
                h_0,
                g_0,
            ) = Helpers.get_standard_values_1d_box(0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
                h_1,
                g_1,
            ) = Helpers.get_standard_values_1d_box(3, dc_decomp)
        else:
            (
                x_0,
                y_0,
                z_0,
                u_c_0,
                w_u_0,
                b_u_0,
                l_c_0,
                w_l_0,
                b_l_0,
            ) = Helpers.get_standard_values_1d_box(0, dc_decomp)
            (
                x_1,
                y_1,
                z_1,
                u_c_1,
                w_u_1,
                b_u_1,
                l_c_1,
                w_l_1,
                b_l_1,
            ) = Helpers.get_standard_values_1d_box(1, dc_decomp)

        if not odd:
            # output (x_0+x_1, x_0+2*x_0) (x_0, x_1)
            x_ = np.concatenate([x_0, x_1], -1)
            z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
            z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
            z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
            y_ = np.concatenate([y_0, y_1], -1)
            b_u_ = np.concatenate([b_u_0, b_u_1], -1)
            u_c_ = np.concatenate([u_c_0, u_c_1], -1)
            b_l_ = np.concatenate([b_l_0, b_l_1], -1)
            l_c_ = np.concatenate([l_c_0, l_c_1], -1)

            if dc_decomp:
                h_ = np.concatenate([h_0, h_1], -1)
                g_ = np.concatenate([g_0, g_1], -1)

            w_u_ = np.zeros((len(x_), 2, 2))
            w_u_[:, 0, 0] = w_u_0[:, 0, 0]
            w_u_[:, 1, 1] = w_u_1[:, 0, 0]

            w_l_ = np.zeros((len(x_), 2, 2))
            w_l_[:, 0, 0] = w_l_0[:, 0, 0]
            w_l_[:, 1, 1] = w_l_1[:, 0, 0]

        else:
            if dc_decomp:
                (
                    x_2,
                    y_2,
                    z_2,
                    u_c_2,
                    w_u_2,
                    b_u_2,
                    l_c_2,
                    w_l_2,
                    b_l_2,
                    h_2,
                    g_2,
                ) = Helpers.get_standard_values_1d_box(2, dc_decomp)
            else:
                (x_2, y_2, z_2, u_c_2, w_u_2, b_u_2, l_c_2, w_l_2, b_l_2) = Helpers.get_standard_values_1d_box(
                    2, dc_decomp
                )
            x_ = np.concatenate([x_0, x_1, x_2], -1)
            z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0], z_2[:, 0]], -1)
            z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1], z_2[:, 1]], -1)
            z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
            y_ = np.concatenate([y_0, y_1, y_2], -1)
            b_u_ = np.concatenate([b_u_0, b_u_1, b_u_2], -1)
            b_l_ = np.concatenate([b_l_0, b_l_1, b_l_2], -1)
            u_c_ = np.concatenate([u_c_0, u_c_1, u_c_2], -1)
            l_c_ = np.concatenate([l_c_0, l_c_1, l_c_2], -1)

            if dc_decomp:
                h_ = np.concatenate([h_0, h_1, h_2], -1)
                g_ = np.concatenate([g_0, g_1, g_2], -1)

            w_u_ = np.zeros((len(x_), 3, 3))
            w_u_[:, 0, 0] = w_u_0[:, 0, 0]
            w_u_[:, 1, 1] = w_u_1[:, 0, 0]
            w_u_[:, 2, 2] = w_u_2[:, 0, 0]

            w_l_ = np.zeros((len(x_), 3, 3))
            w_l_[:, 0, 0] = w_l_0[:, 0, 0]
            w_l_[:, 1, 1] = w_l_1[:, 0, 0]
            w_l_[:, 2, 2] = w_l_2[:, 0, 0]

        if dc_decomp:
            return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, h_, g_]
        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def assert_output_properties_box_nodc(x_, y_, x_min_, x_max_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_, decimal=4):
        assert np.min(x_min_ <= x_max_), "x_min >x_max"

        assert_almost_equal(np.clip(x_min_ - x_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_min >x_")
        assert_almost_equal(np.clip(x_ - x_max_, 0, np.inf), 0.0, decimal=decimal, err_msg="x_max < x_")

        x_expand = x_ + np.zeros_like(x_)
        n_expand = len(w_u_.shape) - len(x_expand.shape)
        for i in range(n_expand):
            x_expand = np.expand_dims(x_expand, -1)

        lower_ = np.sum(w_l_ * x_expand, 1) + b_l_
        upper_ = np.sum(w_u_ * x_expand, 1) + b_u_

        # check that the functions h_ and g_ remains monotonic

        assert_almost_equal(
            np.clip(l_c_ - y_, 0.0, np.inf),
            np.zeros_like(y_),
            decimal=decimal,
            err_msg="l_c >y",
        )
        assert_almost_equal(
            np.clip(y_ - u_c_, 0.0, 1e6),
            np.zeros_like(y_),
            decimal=decimal,
            err_msg="u_c <y",
        )

        #

        assert_almost_equal(
            np.clip(lower_ - y_, 0.0, np.inf),
            np.zeros_like(y_),
            decimal=decimal,
            err_msg="lower_ >y",
        )
        assert_almost_equal(
            np.clip(y_ - upper_, 0.0, 1e6),
            np.zeros_like(y_),
            decimal=decimal,
            err_msg="upper <y",
        )

        # computer lower bounds on the domain

        x_expand_min = x_min_ + np.zeros_like(x_)
        x_expand_max = x_max_ + np.zeros_like(x_)
        n_expand = len(w_u_.shape) - len(x_expand_min.shape)
        for i in range(n_expand):
            x_expand_min = np.expand_dims(x_expand_min, -1)
            x_expand_max = np.expand_dims(x_expand_max, -1)

        lower_ = np.sum(np.maximum(0, w_l_) * x_expand_min, 1) + np.sum(np.minimum(0, w_l_) * x_expand_max, 1) + b_l_
        upper_ = np.sum(np.maximum(0, w_u_) * x_expand_max, 1) + np.sum(np.minimum(0, w_u_) * x_expand_min, 1) + b_u_

        assert_almost_equal(
            np.clip(lower_.min(0) - l_c_.max(0), 0.0, np.inf),
            np.zeros_like(y_.min(0)),
            decimal=decimal,
            err_msg="lower_ >l_c",
        )
        assert_almost_equal(
            np.clip(u_c_.min(0) - upper_.max(0), 0.0, 1e6),
            np.zeros_like(y_.min(0)),
            decimal=decimal,
            err_msg="upper <u_c",
        )

    @staticmethod
    def toy_network_tutorial(dtype="float32"):
        layers = []
        layers.append(Dense(100, input_dim=1, dtype=dtype))  # specify the dimension of the input space
        layers.append(Activation("relu", dtype=dtype))
        layers.append(Dense(100, dtype=dtype))
        layers.append(Dense(1, activation="linear", dtype=dtype))
        model = Sequential(layers)
        return model

    @staticmethod
    def toy_network_tutorial_with_embedded_activation(dtype="float32"):
        layers = []
        layers.append(
            Dense(100, input_dim=1, activation="relu", dtype=dtype)
        )  # specify the dimension of the input space
        layers.append(Dense(100, dtype=dtype))
        layers.append(Dense(1, activation="linear", dtype=dtype))
        model = Sequential(layers)
        return model

    @staticmethod
    def toy_embedded_sequential(dtype="float32"):
        layers = []
        units = 10
        layers.append(
            Dense(units, input_dim=1, activation="relu", dtype=dtype)
        )  # specify the dimension of the input space
        layers.append(
            Helpers.dense_NN_1D(
                dtype=dtype, archi=[2, 3, 2], sequential=True, input_dim=units, activation="relu", use_bias=False
            )
        )
        layers.append(Dense(1, activation="linear", dtype=dtype))
        model = Sequential(layers)
        return model

    @staticmethod
    def dense_NN_1D(input_dim, archi, sequential, activation, use_bias, dtype="float32"):
        layers = [Dense(archi[0], use_bias=use_bias, activation=activation, input_dim=input_dim, dtype=dtype)]
        layers += [Dense(n_i, use_bias=use_bias, activation=activation, dtype=dtype) for n_i in archi[1:]]

        if sequential:
            return Sequential(layers)
        else:
            x = Input(input_dim, dtype=dtype)
            output = layers[0](x)
            for layer_ in layers[1:]:
                output = layer_(output)
            return Model(x, output)

    @staticmethod
    def toy_struct_v0_1D(input_dim, archi, activation, use_bias, merge_op=Add, dtype="float32"):
        nnet_0 = Helpers.dense_NN_1D(
            input_dim=input_dim, archi=archi, sequential=False, activation=activation, use_bias=use_bias, dtype=dtype
        )
        nnet_1 = Dense(archi[-1], use_bias=use_bias, activation="linear", input_dim=input_dim, name="toto", dtype=dtype)

        x = Input(input_dim, dtype=dtype)
        h_0 = nnet_0(x)
        h_1 = nnet_1(x)

        y = merge_op(dtype=dtype)([h_0, h_1])

        return Model(x, y)

    @staticmethod
    def toy_struct_v1_1D(input_dim, archi, sequential, activation, use_bias, merge_op=Add, dtype="float32"):
        nnet_0 = Helpers.dense_NN_1D(
            input_dim=input_dim,
            archi=archi,
            sequential=sequential,
            activation=activation,
            use_bias=use_bias,
            dtype=dtype,
        )

        x = Input(input_dim, dtype=dtype)
        h_0 = nnet_0(x)
        h_1 = nnet_0(x)
        y = merge_op(dtype=dtype)([h_0, h_1])

        return Model(x, y)

    @staticmethod
    def toy_struct_v2_1D(input_dim, archi, sequential, activation, use_bias, merge_op=Add, dtype="float32"):
        nnet_0 = Helpers.dense_NN_1D(
            input_dim=input_dim,
            archi=archi,
            sequential=sequential,
            activation=activation,
            use_bias=use_bias,
            dtype=dtype,
        )
        nnet_1 = Helpers.dense_NN_1D(
            input_dim=input_dim,
            archi=archi,
            sequential=sequential,
            activation=activation,
            use_bias=use_bias,
            dtype=dtype,
        )
        nnet_2 = Dense(archi[-1], use_bias=use_bias, activation="linear", input_dim=input_dim, dtype=dtype)

        x = Input(input_dim, dtype=dtype)
        nnet_0(x)
        nnet_1(x)
        nnet_1.set_weights([-p for p in nnet_0.get_weights()])  # be sure that the produced output will differ
        h_0 = nnet_2(nnet_0(x))
        h_1 = nnet_2(nnet_1(x))
        y = merge_op(dtype=dtype)([h_0, h_1])

        return Model(x, y)

    @staticmethod
    def toy_struct_cnn(dtype="float32", image_data_shape=(6, 6, 2)):
        input_dim = np.prod(image_data_shape)
        layers = [
            Reshape(target_shape=image_data_shape, input_dim=input_dim),
            Conv2D(
                10,
                kernel_size=(3, 3),
                activation="relu",
                data_format="channels_last",
                dtype=dtype,
            ),
            Flatten(dtype=dtype),
            Dense(1, dtype=dtype),
        ]
        return Sequential(layers)

    @staticmethod
    def toy_model(model_name, dtype="float32"):
        if model_name == "tutorial":
            return Helpers.toy_network_tutorial(dtype=dtype)
        elif model_name == "tutorial_activation_embedded":
            return Helpers.toy_network_tutorial_with_embedded_activation(dtype=dtype)
        elif model_name == "merge_v0":
            return Helpers.toy_struct_v0_1D(dtype=dtype, input_dim=1, archi=[2, 3, 2], activation="relu", use_bias=True)
        elif model_name == "merge_v1":
            return Helpers.toy_struct_v1_1D(
                dtype=dtype, input_dim=1, archi=[2, 3, 2], activation="relu", use_bias=True, sequential=False
            )
        elif model_name == "merge_v1_seq":
            return Helpers.toy_struct_v1_1D(
                dtype=dtype, input_dim=1, archi=[2, 3, 2], activation="relu", use_bias=True, sequential=True
            )
        elif model_name == "merge_v1_2":
            return Helpers.toy_struct_v1_1D(
                dtype=dtype, input_dim=2, archi=[2, 3, 2], activation="relu", use_bias=True, sequential=False
            )
        elif model_name == "merge_v2":
            return Helpers.toy_struct_v2_1D(
                dtype=dtype, input_dim=1, archi=[2, 3, 2], activation="relu", use_bias=True, sequential=False
            )
        elif model_name == "cnn":
            return Helpers.toy_struct_cnn(dtype=dtype)
        elif model_name == "embedded_model":
            return Helpers.toy_embedded_sequential(dtype=dtype)
        else:
            raise ValueError(f"model_name {model_name} unknown")


@pytest.fixture
def helpers():
    return Helpers


@pytest.fixture(
    params=[
        "tutorial",
        "tutorial_activation_embedded",
        "merge_v0",
        "merge_v1",
        "merge_v1_seq",
        "merge_v1_2",
        "merge_v2",
        "cnn",
        "embedded_model",
    ]
)
def toy_model(request, helpers):
    model_name = request.param
    return helpers.toy_model(model_name, dtype=K.floatx())


@pytest.fixture(
    params=[
        "tutorial",
        "tutorial_activation_embedded",
        "merge_v0",
        "merge_v1",
        "merge_v1_seq",
        "merge_v2",
        "embedded_model",
    ]
)
def toy_model_1d(request, helpers):
    model_name = request.param
    return helpers.toy_model(model_name, dtype=K.floatx())
