from typing import Union

import keras
import keras.config as keras_config
import keras.ops as K
import numpy as np
import pytest
from keras import KerasTensor, Model
from keras.layers import Input
from pytest_cases import param_fixture, param_fixtures

from decomon.core import BoxDomain, Propagation, Slope
from decomon.keras_utils import (
    BACKEND_JAX,
    BACKEND_NUMPY,
    BACKEND_PYTORCH,
    BACKEND_TENSORFLOW,
    batch_multid_dot,
)
from decomon.models.utils import ConvertMethod
from decomon.types import BackendTensor, Tensor

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
use_bias = param_fixture("use_bias", [True, False])
randomize = param_fixture("randomize", [True, False])
padding = param_fixture("padding", ["same", "valid"])
activation = param_fixture("activation", [None, "relu"])
data_format = param_fixture("data_format", ["channels_last", "channels_first"])
method = param_fixture("method", [m.value for m in ConvertMethod])


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
    def generate_random_tensor(shape_wo_batchsize, batchsize=10, dtype="float32"):
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
        x_shape = perturbation_domain.get_x_input_shape_wo_batchsize(model_input_shape)
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

        return [affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, x_shape]

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
        decomon_input_shape = Helpers.get_decomon_input_shapes(
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
        affine_bounds_to_propagate_shape, constant_oracle_bounds_shape, x_shape = decomon_input_shape
        x = Input(x_shape)
        constant_oracle_bounds = [Input(shape) for shape in constant_oracle_bounds_shape]
        if nobatch:
            affine_bounds_to_propagate = [Input(batch_shape=shape) for shape in affine_bounds_to_propagate_shape]
        else:
            affine_bounds_to_propagate = [Input(shape=shape) for shape in affine_bounds_to_propagate_shape]
        return [affine_bounds_to_propagate, constant_oracle_bounds, x]

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
        if isinstance(perturbation_domain, BoxDomain):
            x = K.repeat(keras_input[:, None], 2, axis=1)
        else:
            raise NotImplementedError

        if affine and not empty:
            batchsize = keras_input.shape[0]
            if propagation == Propagation.FORWARD:
                bias_shape = keras_input.shape[1:]
            else:
                bias_shape = layer_output_shape
            flatten_bias_dim = int(np.prod(bias_shape))
            if diag:
                w_in = K.ones(bias_shape)
            else:
                w_in = K.reshape(K.eye(flatten_bias_dim), bias_shape + bias_shape)
            b_in = K.zeros(bias_shape)
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

        return [affine_bounds_to_propagate, constant_oracle_bounds, x]

    @staticmethod
    def assert_decomon_outputs_equal(output_1, output_2, decimal=5):
        names = [["w_l", "b_l", "w_u", "b_u"], ["l_c", "u_c"]]
        assert len(output_1) == len(output_2)
        for i in range(len(output_1)):
            sub_output_1 = output_1[i]
            sub_output_2 = output_2[i]
            assert len(sub_output_1) == len(sub_output_2)
            for j in range(len(sub_output_1)):
                Helpers.assert_almost_equal(
                    sub_output_1[j],
                    sub_output_2[j],
                    decimal=decimal,
                    err_msg=names[i][j],
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
        decomon_output, keras_output, keras_input, decimal=5
    ):
        affine = len(decomon_output[0]) > 0
        ibp = len(decomon_output) > 1 and len(decomon_output[1]) > 0

        if affine:
            w_l, b_l, w_u, b_u = decomon_output[0]
            lower_affine = batch_multid_dot(keras_input, w_l) + b_l
            upper_affine = batch_multid_dot(keras_input, w_u) + b_u
            Helpers.assert_ordered(lower_affine, keras_output, decimal=decimal, err_msg="lower_affine not ok")
            Helpers.assert_ordered(keras_output, upper_affine, decimal=decimal, err_msg="upper_affine not ok")

        if ibp:
            lower_ibp, upper_ibp = decomon_output[1]
            Helpers.assert_ordered(lower_ibp, keras_output, decimal=decimal, err_msg="lower_ibp not ok")
            Helpers.assert_ordered(keras_output, upper_ibp, decimal=decimal, err_msg="upper_ibp not ok")

    @staticmethod
    def assert_decomon_output_lower_equal_upper(decomon_output, decimal=5):
        affine = len(decomon_output[0]) > 0
        ibp = len(decomon_output) > 1 and len(decomon_output[1]) > 0

        if affine:
            w_l, b_l, w_u, b_u = decomon_output[0]
            Helpers.assert_almost_equal(w_l, w_u, decimal=decimal)
            Helpers.assert_almost_equal(b_l, b_u, decimal=decimal)

        if ibp:
            lower_ibp, upper_ibp = decomon_output[1]
            Helpers.assert_almost_equal(lower_ibp, upper_ibp, decimal=decimal)

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
