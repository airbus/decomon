from typing import Optional, Union

import keras
import keras.config as keras_config
import keras.ops as K
import numpy as np
import pytest
from keras import KerasTensor, Model
from keras.layers import Input
from pytest_cases import (
    fixture,
    fixture_union,
    param_fixture,
    param_fixtures,
    unpack_fixture,
)

from decomon.core import BoxDomain, InputsOutputsSpec, Propagation, Slope
from decomon.keras_utils import (
    BACKEND_JAX,
    BACKEND_NUMPY,
    BACKEND_PYTORCH,
    BACKEND_TENSORFLOW,
    batch_multid_dot,
)
from decomon.models.utils import ConvertMethod
from decomon.types import BackendTensor, Tensor

empty, diag, nobatch = param_fixtures(
    "empty, diag, nobatch",
    [
        (True, True, True),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
    ids=["identity", "diagonal-nobatch", "diagonal", "nobatch", "generic"],
)
ibp, affine, propagation = param_fixtures(
    "ibp, affine, propagation",
    [
        (True, True, Propagation.FORWARD),
        (False, True, Propagation.FORWARD),
        (True, False, Propagation.FORWARD),
        (True, True, Propagation.BACKWARD),
    ],
    ids=["forward-hybrid", "forward-affine", "forward-ibp", "backward"],
)
slope = param_fixture("slope", [s.value for s in Slope])
n = param_fixture("n", list(range(10)))
odd = param_fixture("odd", list(range(2)))
use_bias = param_fixture("use_bias", [True, False])
randomize = param_fixture("randomize", [True, False])
padding = param_fixture("padding", ["same", "valid"])
activation = param_fixture("activation", [None, "relu"])
data_format = param_fixture("data_format", ["channels_last", "channels_first"])
method = param_fixture("method", [m.value for m in ConvertMethod])
input_shape = param_fixture("input_shape", [(1,), (3,), (5, 2, 3)], ids=["0d", "1d", "multid"])


@pytest.fixture
def batchsize():
    return 10


@pytest.fixture
def perturbation_domain():
    return BoxDomain()


@pytest.fixture(params=[32, 64, 16])
def floatx(request):
    # setup
    eps_bak = keras_config.epsilon()  # store current epsilon
    floatx_bak = keras_config.floatx()  # store current floatx
    precision = request.param
    keras_config.set_floatx("float{}".format(precision))
    if precision == 16:
        keras_config.set_epsilon(1e-2)
    # actual value
    yield precision
    # tear down
    keras_config.set_epsilon(eps_bak)
    keras_config.set_floatx(floatx_bak)


@pytest.fixture
def decimal(floatx):
    if floatx == 16:
        return 2
    else:
        return 4


class ModelNumpyFromKerasTensors:
    def __init__(self, inputs: list[KerasTensor], outputs: list[KerasTensor]):
        self.inputs = inputs
        self.outputs = outputs
        self._model = Model(inputs, outputs)

    def __call__(self, inputs_: list[np.ndarray]):
        output_tensors = self._model(inputs_)
        if isinstance(output_tensors, list):
            return [K.convert_to_numpy(output) for output in output_tensors]
        else:
            return K.convert_to_numpy(output_tensors)


class Helpers:
    function = ModelNumpyFromKerasTensors

    @staticmethod
    def in_GPU_mode() -> bool:
        backend = keras.config.backend()
        if backend == BACKEND_TENSORFLOW:
            import tensorflow

            return len(tensorflow.config.list_physical_devices("GPU")) > 0
        elif backend == BACKEND_PYTORCH:
            import torch

            return torch.cuda.is_available()
        elif backend == BACKEND_NUMPY:
            return False
        elif backend == BACKEND_JAX:
            import jax

            return jax.devices()[0].platform != "cpu"
        else:
            raise NotImplementedError(f"Not implemented for {backend} backend.")

    @staticmethod
    def generate_random_tensor(shape_wo_batchsize, batchsize=10, dtype=keras_config.floatx()):
        shape = (batchsize,) + shape_wo_batchsize
        return K.convert_to_tensor(np.random.random(shape), dtype=dtype)

    @staticmethod
    def get_decomon_input_shapes(
        model_input_shape,
        model_output_shape,
        layer_input_shape,
        layer_output_shape,
        ibp,
        affine,
        propagation,
        perturbation_domain,
        empty=False,
        diag=False,
        nobatch=False,
    ):
        if affine and propagation == Propagation.FORWARD:
            model_inputs_shape = [perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)]
        else:
            model_inputs_shape = []
        if affine and not empty:
            if propagation == Propagation.FORWARD:
                b_in_shape = layer_input_shape
                w_in_shape = model_input_shape + layer_input_shape
            else:
                b_in_shape = model_output_shape
                w_in_shape = layer_output_shape + model_output_shape
            if diag:
                w_in_shape = b_in_shape

            affine_bounds_to_propagate_shape = [w_in_shape, b_in_shape, w_in_shape, b_in_shape]
        else:
            affine_bounds_to_propagate_shape = []
        if ibp:
            constant_oracle_bounds_shape = [layer_input_shape, layer_input_shape]
        else:
            constant_oracle_bounds_shape = []

        return affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, model_inputs_shape

    @staticmethod
    def get_decomon_symbolic_inputs(
        model_input_shape,
        model_output_shape,
        layer_input_shape,
        layer_output_shape,
        ibp,
        affine,
        propagation,
        perturbation_domain,
        empty=False,
        diag=False,
        nobatch=False,
        dtype=keras_config.floatx(),
    ):
        """Generate decomon symbolic inputs for a decomon layer

        To be used as `decomon_layer(*decomon_inputs)`.

        Args:
            model_input_shape:
            model_output_shape:
            layer_input_shape:
            layer_output_shape:
            ibp:
            affine:
            propagation:
            perturbation_domain:

        Returns:

        """
        (
            affine_bounds_to_propagate_shape,
            constant_oracle_bounds_shape,
            model_inputs_shape,
        ) = Helpers.get_decomon_input_shapes(
            model_input_shape,
            model_output_shape,
            layer_input_shape,
            layer_output_shape,
            ibp,
            affine,
            propagation,
            perturbation_domain,
            empty=empty,
            diag=diag,
            nobatch=nobatch,
        )
        model_inputs = [Input(shape, dtype=dtype) for shape in model_inputs_shape]
        constant_oracle_bounds = [Input(shape, dtype=dtype) for shape in constant_oracle_bounds_shape]
        if nobatch:
            affine_bounds_to_propagate = [
                Input(batch_shape=shape, dtype=dtype) for shape in affine_bounds_to_propagate_shape
            ]
        else:
            affine_bounds_to_propagate = [Input(shape=shape, dtype=dtype) for shape in affine_bounds_to_propagate_shape]
        inputs_outputs_spec = InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            perturbation_domain=perturbation_domain,
            layer_input_shape=layer_input_shape,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
        )
        return inputs_outputs_spec.flatten_inputs(
            affine_bounds_to_propagate=affine_bounds_to_propagate,
            constant_oracle_bounds=constant_oracle_bounds,
            model_inputs=model_inputs,
        )

    @staticmethod
    def generate_simple_decomon_layer_inputs_from_keras_input(
        keras_input,
        layer_output_shape,
        ibp,
        affine,
        propagation,
        perturbation_domain,
        empty=False,
        diag=False,
        nobatch=False,
        dtype=keras_config.floatx(),
    ):
        """Generate simple decomon inputs for a layer from the corresponding keras input

        Hypothesis: single-layer model => model input/output = layer input/output

        For affine bounds, weights= identity + bias = 0
        For constant bounds, (keras_input, keras_input)

        To be used as `decomon_layer(*decomon_inputs)`.

        Args:
            keras_input:
            layer_output_shape:
            ibp:
            affine:
            propagation:
            perturbation_domain:
            empty:
            diag:
            nobatch:

        Returns:

        """
        layer_input_shape = keras_input.shape[1:]
        model_input_shape = layer_input_shape
        model_output_shape = layer_output_shape
        if affine and propagation == Propagation.FORWARD:
            if isinstance(perturbation_domain, BoxDomain):
                x = K.repeat(keras_input[:, None], 2, axis=1)
            else:
                raise NotImplementedError
            model_inputs = [x]
        else:
            model_inputs = []

        if affine and not empty:
            batchsize = keras_input.shape[0]
            if propagation == Propagation.FORWARD:
                bias_shape = layer_input_shape
            else:
                bias_shape = layer_output_shape
            flatten_bias_dim = int(np.prod(bias_shape))
            if diag:
                w_in = K.ones(bias_shape, dtype=dtype)
            else:
                w_in = K.reshape(K.eye(flatten_bias_dim, dtype=dtype), bias_shape + bias_shape)
            b_in = K.zeros(bias_shape, dtype=dtype)
            if not nobatch:
                w_in = K.repeat(
                    w_in[None],
                    batchsize,
                    axis=0,
                )
                b_in = K.repeat(
                    b_in[None],
                    batchsize,
                    axis=0,
                )
            affine_bounds_to_propagate = [w_in, b_in, w_in, b_in]
        else:
            affine_bounds_to_propagate = []

        if ibp:
            constant_oracle_bounds = [keras_input, keras_input]
        else:
            constant_oracle_bounds = []

        inputs_outputs_spec = InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            perturbation_domain=perturbation_domain,
            layer_input_shape=layer_input_shape,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
        )
        return inputs_outputs_spec.flatten_inputs(
            affine_bounds_to_propagate=affine_bounds_to_propagate,
            constant_oracle_bounds=constant_oracle_bounds,
            model_inputs=model_inputs,
        )

    @staticmethod
    def get_standard_values_0d_box(n, batchsize=10):
        """A set of functions with their monotonic decomposition for testing the activations"""
        w_u_ = np.ones(batchsize)
        b_u_ = np.zeros(batchsize)
        w_l_ = np.ones(batchsize)
        b_l_ = np.zeros(batchsize)

        if n == 0:
            # identity
            y_ = np.linspace(-2, -1, batchsize)
            x_ = np.linspace(-2, -1, batchsize)

        elif n == 1:
            y_ = np.linspace(1, 2, batchsize)
            x_ = np.linspace(1, 2, batchsize)

        elif n == 2:
            y_ = np.linspace(-1, 1, batchsize)
            x_ = np.linspace(-1, 1, batchsize)

        elif n == 3:
            # identity
            y_ = np.linspace(-2, -1, batchsize)
            x_ = np.linspace(-2, -1, batchsize)

        elif n == 4:
            y_ = np.linspace(1, 2, batchsize)
            x_ = np.linspace(1, 2, batchsize)

        elif n == 5:
            y_ = np.linspace(-1, 1, batchsize)
            x_ = np.linspace(-1, 1, batchsize)

        elif n == 6:
            # cosine function
            x_ = np.linspace(-np.pi, np.pi, batchsize)
            y_ = np.cos(x_)
            w_u_ = np.zeros_like(x_)
            w_l_ = np.zeros_like(x_)
            b_u_ = np.ones_like(x_)
            b_l_ = -np.ones_like(x_)

        elif n == 7:
            # h and g >0
            h_ = np.linspace(0.5, 2, batchsize)
            g_ = np.linspace(1, 2, batchsize)[::-1]
            x_ = h_ + g_
            y_ = h_ + g_

        elif n == 8:
            # h <0 and g <0
            # h_max+g_max <=0
            h_ = np.linspace(-2, -1, batchsize)
            g_ = np.linspace(-2, -1, batchsize)[::-1]
            y_ = h_ + g_
            x_ = h_ + g_

        elif n == 9:
            # h >0 and g <0
            # h_min+g_min >=0
            h_ = np.linspace(4, 5, batchsize)
            g_ = np.linspace(-2, -1, batchsize)[::-1]
            y_ = h_ + g_
            x_ = h_ + g_

        else:
            raise ValueError("n must be between 0 and 9.")

        x_min_ = x_.min() + np.zeros_like(x_)
        x_max_ = x_.max() + np.zeros_like(x_)

        x_0_ = np.concatenate([x_min_[:, None], x_max_[:, None]], 1)

        u_c_ = np.max(y_) * np.ones((batchsize,))
        l_c_ = np.min(y_) * np.ones((batchsize,))

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
        return [e.astype(keras_config.floatx()) for e in output]

    @staticmethod
    def get_tensor_decomposition_0d_box():
        return [
            Input((1,), dtype=keras_config.floatx()),
            Input((1,), dtype=keras_config.floatx()),
            Input((2, 1), dtype=keras_config.floatx()),
            Input((1,), dtype=keras_config.floatx()),
            Input((1, 1), dtype=keras_config.floatx()),
            Input((1,), dtype=keras_config.floatx()),
            Input((1,), dtype=keras_config.floatx()),
            Input((1, 1), dtype=keras_config.floatx()),
            Input((1,), dtype=keras_config.floatx()),
        ]

    @staticmethod
    def get_standard_values_1d_box(odd=1, batchsize=10):
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
        ) = Helpers.get_standard_values_0d_box(n=0, batchsize=batchsize)
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
        ) = Helpers.get_standard_values_0d_box(n=1, batchsize=batchsize)

        if not odd:
            # output (x_0+x_1, x_0+2*x_1)
            x_ = np.concatenate([x_0, x_1], -1)
            z_min_ = np.concatenate([z_0[:, 0], z_1[:, 0]], -1)
            z_max_ = np.concatenate([z_0[:, 1], z_1[:, 1]], -1)
            z_ = np.concatenate([z_min_[:, None], z_max_[:, None]], 1)
            y_ = np.concatenate([y_0 + y_1, y_0 + 2 * y_1], -1)
            b_u_ = np.concatenate([b_u_0 + b_u_1, b_u_0 + 2 * b_u_1], -1)
            u_c_ = np.concatenate([u_c_0 + u_c_1, u_c_0 + 2 * u_c_1], -1)
            b_l_ = np.concatenate([b_l_0 + b_l_1, b_l_0 + 2 * b_l_1], -1)
            l_c_ = np.concatenate([l_c_0 + l_c_1, l_c_0 + 2 * l_c_1], -1)

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
            ) = Helpers.get_standard_values_0d_box(n=2, batchsize=batchsize)

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

        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def get_tensor_decomposition_1d_box(odd=1):
        n = Helpers.get_input_dim_1d_box(odd)

        return [
            Input((n,), dtype=keras_config.floatx()),
            Input((n,), dtype=keras_config.floatx()),
            Input((2, n), dtype=keras_config.floatx()),
            Input((n,), dtype=keras_config.floatx()),
            Input((n, n), dtype=keras_config.floatx()),
            Input((n,), dtype=keras_config.floatx()),
            Input((n,), dtype=keras_config.floatx()),
            Input((n, n), dtype=keras_config.floatx()),
            Input((n,), dtype=keras_config.floatx()),
        ]

    @staticmethod
    def get_input_dim_1d_box(odd):
        if odd:
            return 3
        else:
            return 2

    @staticmethod
    def get_standard_values_images_box(data_format="channels_last", odd=0, m0=0, m1=1, batchsize=10):
        if data_format == "channels_last":
            output = Helpers.build_image_from_2D_box(odd=odd, m0=m0, m1=m1, batchsize=batchsize)
            x_0, y_0, z_0, u_c_0, w_u_0, b_u_0, l_c_0, w_l_0, b_l_0 = output

            x_ = x_0
            z_ = z_0

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

        else:
            output = Helpers.get_standard_values_images_box(
                data_format="channels_last", odd=odd, m0=m0, m1=m1, batchsize=batchsize
            )

            x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_ = output
            y_ = np.transpose(y_, (0, 3, 1, 2))
            u_c_ = np.transpose(u_c_, (0, 3, 1, 2))
            l_c_ = np.transpose(l_c_, (0, 3, 1, 2))
            b_u_ = np.transpose(b_u_, (0, 3, 1, 2))
            b_l_ = np.transpose(b_l_, (0, 3, 1, 2))
            w_u_ = np.transpose(w_u_, (0, 1, 4, 2, 3))
            w_l_ = np.transpose(w_l_, (0, 1, 4, 2, 3))

        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def get_tensor_decomposition_images_box(data_format, odd):
        n = Helpers.get_input_dim_images_box(odd)

        if data_format == "channels_last":
            # x, y, z, u, w_u, b_u, l, w_l, b_l

            output = [
                Input((2,), dtype=keras_config.floatx()),
                Input((n, n, 2), dtype=keras_config.floatx()),
                Input((2, 2), dtype=keras_config.floatx()),
                Input((n, n, 2), dtype=keras_config.floatx()),
                Input((2, n, n, 2), dtype=keras_config.floatx()),
                Input((n, n, 2), dtype=keras_config.floatx()),
                Input((n, n, 2), dtype=keras_config.floatx()),
                Input((2, n, n, 2), dtype=keras_config.floatx()),
                Input((n, n, 2), dtype=keras_config.floatx()),
            ]

        else:
            output = [
                Input((2,), dtype=keras_config.floatx()),
                Input((2, n, n), dtype=keras_config.floatx()),
                Input((2, 2), dtype=keras_config.floatx()),
                Input((2, n, n), dtype=keras_config.floatx()),
                Input((2, 2, n, n), dtype=keras_config.floatx()),
                Input((2, n, n), dtype=keras_config.floatx()),
                Input((2, n, n), dtype=keras_config.floatx()),
                Input((2, 2, n, n), dtype=keras_config.floatx()),
                Input((2, n, n), dtype=keras_config.floatx()),
            ]

        return output

    @staticmethod
    def get_input_dim_images_box(odd):
        if odd:
            return 7
        else:
            return 6

    @staticmethod
    def build_image_from_2D_box(odd=0, m0=0, m1=1, batchsize=10):
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
        ) = Helpers.build_image_from_1D_box(odd=odd, m=m0, batchsize=batchsize)
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
        ) = Helpers.build_image_from_1D_box(odd=odd, m=m1, batchsize=batchsize)

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

        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def build_image_from_1D_box(odd=0, m=0, batchsize=10):
        n = Helpers.get_input_dim_images_box(odd)

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
        ) = Helpers.get_standard_values_0d_box(n=m, batchsize=batchsize)

        y_ = np.concatenate([(i + 1) * y_0 for i in range(n * n)], -1).reshape((-1, n, n))
        b_u_ = np.concatenate([(i + 1) * b_u_0 for i in range(n * n)], -1).reshape((-1, n, n))
        b_l_ = np.concatenate([(i + 1) * b_l_0 for i in range(n * n)], -1).reshape((-1, n, n))

        u_c_ = np.concatenate([(i + 1) * u_c_0 for i in range(n * n)], -1).reshape((-1, n, n))
        l_c_ = np.concatenate([(i + 1) * l_c_0 for i in range(n * n)], -1).reshape((-1, n, n))

        w_u_ = np.zeros((len(x_), 1, n * n))
        w_l_ = np.zeros((len(x_), 1, n * n))

        for i in range(n * n):
            w_u_[:, 0, i] = (i + 1) * w_u_0[:, 0, 0]
            w_l_[:, 0, i] = (i + 1) * w_l_0[:, 0, 0]

        w_u_ = w_u_.reshape((-1, 1, n, n))
        w_l_ = w_l_.reshape((-1, 1, n, n))

        return [x_, y_, z_, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    @staticmethod
    def assert_decomon_outputs_equal(output_1, output_2, decimal=5):
        assert len(output_1) == len(output_2)
        for i in range(len(output_1)):
            Helpers.assert_almost_equal(
                output_1[i],
                output_2[i],
                decimal=decimal,
            )

    @staticmethod
    def assert_ordered(lower: BackendTensor, upper: BackendTensor, decimal: int = 5, err_msg: str = ""):
        lower_np = K.convert_to_numpy(lower)
        upper_np = K.convert_to_numpy(upper)
        np.testing.assert_almost_equal(
            np.clip(lower_np - upper_np, 0.0, np.inf),
            np.zeros_like(lower_np),
            decimal=decimal,
            err_msg=err_msg,
        )

    @staticmethod
    def assert_almost_equal(x: BackendTensor, y: BackendTensor, decimal: int = 5, err_msg: str = ""):
        np.testing.assert_almost_equal(
            K.convert_to_numpy(x),
            K.convert_to_numpy(y),
            decimal=decimal,
            err_msg=err_msg,
        )

    @staticmethod
    def assert_decomon_output_compare_with_keras_input_output_single_layer(
        decomon_output, keras_output, keras_input, ibp, affine, propagation, decimal=5
    ):
        Helpers.assert_decomon_output_compare_with_keras_input_output_layer(
            decomon_output,
            keras_layer_output=keras_output,
            keras_layer_input=keras_input,
            keras_model_input=keras_input,
            keras_model_output=keras_output,
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            decimal=decimal,
        )

    @staticmethod
    def assert_decomon_output_compare_with_keras_input_output_layer(
        decomon_output,
        keras_layer_output,
        keras_layer_input,
        keras_model_input,
        keras_model_output,
        ibp,
        affine,
        propagation,
        decimal=5,
    ):
        inputs_outputs_spec = InputsOutputsSpec(
            ibp=ibp,
            affine=affine,
            propagation=propagation,
            layer_input_shape=keras_layer_input.shape[1:],
            model_output_shape=keras_model_output.shape[1:],
        )
        affine_bounds_propagated, constant_bounds_propagated = inputs_outputs_spec.split_outputs(outputs=decomon_output)

        if propagation == Propagation.FORWARD:
            keras_input = keras_model_input
            keras_output = keras_layer_output
        else:
            keras_input = keras_layer_input
            keras_output = keras_model_output

        if affine or propagation == Propagation.BACKWARD:
            if len(affine_bounds_propagated) == 0:
                # identity case
                lower_affine = keras_input
                upper_affine = keras_input
            else:
                w_l, b_l, w_u, b_u = affine_bounds_propagated
                diagonal = (False, w_l.shape == b_l.shape)
                missing_batchsize = (False, len(b_l.shape) < len(keras_output.shape))
                lower_affine = (
                    batch_multid_dot(keras_input, w_l, diagonal=diagonal, missing_batchsize=missing_batchsize) + b_l
                )
                upper_affine = (
                    batch_multid_dot(keras_input, w_u, diagonal=diagonal, missing_batchsize=missing_batchsize) + b_u
                )
            Helpers.assert_ordered(lower_affine, keras_output, decimal=decimal, err_msg="lower_affine not ok")
            Helpers.assert_ordered(keras_output, upper_affine, decimal=decimal, err_msg="upper_affine not ok")

        if ibp and propagation == Propagation.FORWARD:
            lower_ibp, upper_ibp = constant_bounds_propagated
            Helpers.assert_ordered(lower_ibp, keras_output, decimal=decimal, err_msg="lower_ibp not ok")
            Helpers.assert_ordered(keras_output, upper_ibp, decimal=decimal, err_msg="upper_ibp not ok")

    @staticmethod
    def assert_decomon_output_lower_equal_upper(decomon_output, ibp, affine, propagation, decimal=5):
        inputs_outputs_specs = InputsOutputsSpec(
            ibp=ibp, affine=affine, propagation=propagation, layer_input_shape=tuple(), model_output_shape=tuple()
        )
        affine_bounds_propagated, constant_bounds_propagated = inputs_outputs_specs.split_outputs(
            outputs=decomon_output
        )
        if propagation == Propagation.BACKWARD or affine:
            w_l, b_l, w_u, b_u = affine_bounds_propagated
            Helpers.assert_almost_equal(w_l, w_u, decimal=decimal)
            Helpers.assert_almost_equal(b_l, b_u, decimal=decimal)

        if propagation == Propagation.FORWARD and ibp:
            lower_ibp, upper_ibp = constant_bounds_propagated
            Helpers.assert_almost_equal(lower_ibp, upper_ibp, decimal=decimal)

    @staticmethod
    def replace_none_by_batchsize(shapes: list[tuple[Optional[int], ...]], batchsize: int) -> list[tuple[int]]:
        return [tuple(dim if dim is not None else batchsize for dim in shape) for shape in shapes]

    @staticmethod
    def predict_on_small_numpy(
        model: Model, x: Union[np.ndarray, list[np.ndarray]]
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """Make predictions for model directly on small numpy arrays

        Avoid using `model.predict()` known to be not designed for small arrays,
        and leading to memory leaks when used in loops.

        See https://keras.io/api/models/model_training_apis/#predict-method and
        https://github.com/tensorflow/tensorflow/issues/44711

        Args:
            model:
            x:

        Returns:

        """
        output_tensors = model(x)
        if isinstance(output_tensors, list):
            return [K.convert_to_numpy(output) for output in output_tensors]
        else:
            return K.convert_to_numpy(output_tensors)


@pytest.fixture
def helpers():
    return Helpers


@fixture
def simple_layer_input_functions(
    ibp, affine, propagation, perturbation_domain, batchsize, input_shape, empty, diag, nobatch, helpers
):
    keras_symbolic_model_input_fn = lambda: Input(input_shape)
    keras_symbolic_layer_input_fn = lambda keras_symbolic_model_input: keras_symbolic_model_input

    decomon_symbolic_input_fn = lambda output_shape: helpers.get_decomon_symbolic_inputs(
        model_input_shape=input_shape,
        model_output_shape=output_shape,
        layer_input_shape=input_shape,
        layer_output_shape=output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        empty=empty,
        diag=diag,
        nobatch=nobatch,
    )

    keras_model_input_fn = lambda: helpers.generate_random_tensor(input_shape, batchsize=batchsize)
    keras_layer_input_fn = lambda keras_model_input: keras_model_input

    decomon_input_fn = lambda keras_model_input, keras_layer_input, output_shape: helpers.generate_simple_decomon_layer_inputs_from_keras_input(
        keras_input=keras_layer_input,
        layer_output_shape=output_shape,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        perturbation_domain=perturbation_domain,
        empty=empty,
        diag=diag,
        nobatch=nobatch,
    )

    return (
        keras_symbolic_model_input_fn,
        keras_symbolic_layer_input_fn,
        decomon_symbolic_input_fn,
        keras_model_input_fn,
        keras_layer_input_fn,
        decomon_input_fn,
        True,
    )


def convert_standard_input_functions_for_single_layer(
    get_tensor_decomposition_fn, get_standard_values_fn, ibp, affine, propagation, perturbation_domain, helpers
):
    keras_symbolic_model_input_fn = lambda: get_tensor_decomposition_fn()[0]
    keras_symbolic_layer_input_fn = lambda _: get_tensor_decomposition_fn()[1]
    keras_model_input_fn = lambda: K.convert_to_tensor(get_standard_values_fn()[0])
    keras_layer_input_fn = lambda _: K.convert_to_tensor(get_standard_values_fn()[1])

    if propagation == Propagation.FORWARD:

        def decomon_symbolic_input_fn(output_shape):
            x, y, z, u_c, w_u, b_u, l_c, w_l, b_l = get_tensor_decomposition_fn()
            layer_input_shape = y.shape[1:]
            model_input_shape = x.shape[1:]

            affine_bounds_to_propagate = [w_l, b_l, w_u, b_u]
            constant_oracle_bounds = [l_c, u_c]
            if isinstance(perturbation_domain, BoxDomain):
                model_inputs = [z]
            else:
                raise NotImplementedError

            inputs_outputs_spec = InputsOutputsSpec(
                ibp=ibp,
                affine=affine,
                propagation=propagation,
                perturbation_domain=perturbation_domain,
                layer_input_shape=layer_input_shape,
                model_input_shape=model_input_shape,
                model_output_shape=output_shape,
            )
            return inputs_outputs_spec.flatten_inputs(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                constant_oracle_bounds=constant_oracle_bounds,
                model_inputs=model_inputs,
            )

        def decomon_input_fn(keras_model_input, keras_layer_input, output_shape):
            x, y, z, u_c, w_u, b_u, l_c, w_l, b_l = get_standard_values_fn()
            layer_input_shape = y.shape[1:]
            model_input_shape = x.shape[1:]

            if affine:
                affine_bounds_to_propagate = [K.convert_to_tensor(a) for a in (w_l, b_l, w_u, b_u)]
            else:
                affine_bounds_to_propagate = []
            if ibp:
                constant_oracle_bounds = [K.convert_to_tensor(a) for a in (l_c, u_c)]
            else:
                constant_oracle_bounds = []
            if propagation == Propagation.FORWARD and affine:
                if isinstance(perturbation_domain, BoxDomain):
                    model_inputs = [K.convert_to_tensor(z)]
                else:
                    raise NotImplementedError
            else:
                model_inputs = []

            inputs_outputs_spec = InputsOutputsSpec(
                ibp=ibp,
                affine=affine,
                propagation=propagation,
                perturbation_domain=perturbation_domain,
                layer_input_shape=layer_input_shape,
                model_input_shape=model_input_shape,
                model_output_shape=output_shape,
            )
            return inputs_outputs_spec.flatten_inputs(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                constant_oracle_bounds=constant_oracle_bounds,
                model_inputs=model_inputs,
            )

    else:  # backward

        def decomon_symbolic_input_fn(output_shape):
            x, y, z, u_c, w_u, b_u, l_c, w_l, b_l = get_tensor_decomposition_fn()
            layer_input_shape = y.shape[1:]
            model_input_shape = x.shape[1:]

            if ibp:
                constant_oracle_bounds = [l_c, u_c]
            else:
                constant_oracle_bounds = []
            if propagation == Propagation.FORWARD and affine:
                if isinstance(perturbation_domain, BoxDomain):
                    model_inputs = [z]
                else:
                    raise NotImplementedError
            else:
                model_inputs = []

            inputs_outputs_spec = InputsOutputsSpec(
                ibp=ibp,
                affine=affine,
                propagation=propagation,
                perturbation_domain=perturbation_domain,
                layer_input_shape=layer_input_shape,
                model_input_shape=model_input_shape,
                model_output_shape=output_shape,
            )

            # take identity affine bounds
            if affine:
                simple_decomon_inputs = helpers.get_decomon_symbolic_inputs(
                    model_input_shape=model_input_shape,
                    model_output_shape=output_shape,
                    layer_input_shape=layer_input_shape,
                    layer_output_shape=output_shape,
                    ibp=ibp,
                    affine=affine,
                    propagation=propagation,
                    perturbation_domain=perturbation_domain,
                    empty=empty,
                    diag=diag,
                    nobatch=nobatch,
                )
                affine_bounds_to_propagate, _, _ = inputs_outputs_spec.split_inputs(simple_decomon_inputs)
            else:
                affine_bounds_to_propagate = []

            return inputs_outputs_spec.flatten_inputs(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                constant_oracle_bounds=constant_oracle_bounds,
                model_inputs=model_inputs,
            )

        def decomon_input_fn(keras_model_input, keras_layer_input, output_shape):
            x, y, z, u_c, w_u, b_u, l_c, w_l, b_l = get_standard_values_fn()
            layer_input_shape = y.shape[1:]
            model_input_shape = x.shape[1:]

            if ibp:
                constant_oracle_bounds = [K.convert_to_tensor(a) for a in (l_c, u_c)]
            else:
                constant_oracle_bounds = []
            if propagation == Propagation.FORWARD and affine:
                if isinstance(perturbation_domain, BoxDomain):
                    model_inputs = [K.convert_to_tensor(z)]
                else:
                    raise NotImplementedError
            else:
                model_inputs = []

            inputs_outputs_spec = InputsOutputsSpec(
                ibp=ibp,
                affine=affine,
                propagation=propagation,
                perturbation_domain=perturbation_domain,
                layer_input_shape=layer_input_shape,
                model_input_shape=model_input_shape,
                model_output_shape=output_shape,
            )

            #  take identity affine bounds
            if affine:
                simple_decomon_inputs = helpers.generate_simple_decomon_layer_inputs_from_keras_input(
                    keras_input=keras_layer_input,
                    layer_output_shape=output_shape,
                    ibp=ibp,
                    affine=affine,
                    propagation=propagation,
                    perturbation_domain=perturbation_domain,
                    empty=empty,
                    diag=diag,
                    nobatch=nobatch,
                )
                affine_bounds_to_propagate, _, _ = inputs_outputs_spec.split_inputs(simple_decomon_inputs)
            else:
                affine_bounds_to_propagate = []

            return inputs_outputs_spec.flatten_inputs(
                affine_bounds_to_propagate=affine_bounds_to_propagate,
                constant_oracle_bounds=constant_oracle_bounds,
                model_inputs=model_inputs,
            )

    return (
        keras_symbolic_model_input_fn,
        keras_symbolic_layer_input_fn,
        decomon_symbolic_input_fn,
        keras_model_input_fn,
        keras_layer_input_fn,
        decomon_input_fn,
        False,
    )


@fixture
def standard_layer_input_functions_0d(n, ibp, affine, propagation, batchsize, helpers):
    perturbation_domain = BoxDomain()
    get_tensor_decomposition_fn = helpers.get_tensor_decomposition_0d_box
    get_standard_values_fn = lambda: helpers.get_standard_values_0d_box(n=n, batchsize=batchsize)
    return convert_standard_input_functions_for_single_layer(
        get_tensor_decomposition_fn, get_standard_values_fn, ibp, affine, propagation, perturbation_domain, helpers
    )


@fixture
def standard_layer_input_functions_1d(odd, ibp, affine, propagation, batchsize, helpers):
    perturbation_domain = BoxDomain()
    get_tensor_decomposition_fn = lambda: helpers.get_tensor_decomposition_1d_box(odd=odd)
    get_standard_values_fn = lambda: helpers.get_standard_values_1d_box(odd=odd, batchsize=batchsize)
    return convert_standard_input_functions_for_single_layer(
        get_tensor_decomposition_fn, get_standard_values_fn, ibp, affine, propagation, perturbation_domain, helpers
    )


@fixture
def standard_layer_input_functions_multid(data_format, ibp, affine, propagation, batchsize, helpers):
    perturbation_domain = BoxDomain()
    odd, m0, m1 = 0, 0, 1
    get_tensor_decomposition_fn = lambda: helpers.get_tensor_decomposition_images_box(data_format=data_format, odd=odd)
    get_standard_values_fn = lambda: helpers.get_standard_values_images_box(
        data_format=data_format, odd=odd, m0=m0, m1=m1, batchsize=batchsize
    )
    return convert_standard_input_functions_for_single_layer(
        get_tensor_decomposition_fn, get_standard_values_fn, ibp, affine, propagation, perturbation_domain, helpers
    )


layer_input_functions = fixture_union(
    "layer_input_functions",
    [
        simple_layer_input_functions,
        standard_layer_input_functions_0d,
        standard_layer_input_functions_1d,
        standard_layer_input_functions_multid,
    ],
    unpack_into="keras_symbolic_model_input_fn, keras_symbolic_layer_input_fn, decomon_symbolic_input_fn, keras_model_input_fn, keras_layer_input_fn, decomon_input_fn, equal_bounds",
)

(
    simple_keras_symbolic_model_input_fn,
    simple_keras_symbolic_layer_input_fn,
    simple_decomon_symbolic_input_fn,
    simple_keras_model_input_fn,
    simple_keras_layer_input_fn,
    simple_decomon_input_fn,
    simple_equal_bounds,
) = unpack_fixture(
    "simple_keras_symbolic_model_input_fn, simple_keras_symbolic_layer_input_fn, simple_decomon_symbolic_input_fn, simple_keras_model_input_fn, simple_keras_layer_input_fn, simple_decomon_input_fn, simple_equal_bounds",
    simple_layer_input_functions,
)
