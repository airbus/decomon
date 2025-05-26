# create a general test to compare a layer with auto lirpa
from collections import defaultdict

import keras
import keras.ops as K
import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from jacobinet import clone_to_backward
from keras.layers import Activation, Dense, Reshape
from keras.models import Sequential
from torch import nn
from torch.nn import Linear

from decomon.models.convert import clone


def map_decomon_methods_2_lirpa(method: str):
    if method.lower() == "crown":
        return "CROWN"
    elif method.lower() == "forward-ibp":
        return "IBP"
    elif method.lower() == "crown-forward-ibp":
        return "CROWN-IBP"
    elif method.lower() == "forward-hybrid":
        return "CROWN-IBP"
    elif method.lower() == "forward-affine":
        return "forward"
    else:
        raise ValueError("unrecognized method {}".format(method))


def train_model(keras_model):
    input_dim = keras_model.inputs[0].shape[-1]
    output_dim = keras_model.outputs[0].shape[-1]
    N = 100
    x_train = np.reshape(np.random.randn(N * input_dim), (N, input_dim))
    y_train = np.reshape(np.random.randn(N * output_dim), (N, output_dim))

    keras_model.compile("adam", "mse")
    keras_model.fit(x_train, y_train, epochs=2, verbose=1)


def build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=False, output_dim=2):
    keras_layers = [Dense(np.prod(input_shape)), Reshape(input_shape), keras_layer, Reshape((-1,))]
    if wo_linearity:
        keras_layers += [Dense(output_dim)]
    else:
        keras_layers += [Activation("relu"), Dense(output_dim)]
    keras_model = Sequential(keras_layers)
    _ = keras_model(np.ones((1, input_dim)))

    return keras_model


def build_keras_model_maxpool(keras_layer, input_shape, input_dim, wo_linearity=False):
    keras_layers = [Reshape(input_shape), keras_layer, Reshape((-1,))]
    keras_model = Sequential(keras_layers)
    _ = keras_model(np.ones((1, np.prod(input_shape))))

    return keras_model


def build_keras_model_activation(keras_layer, input_dim, wo_linearity=False):
    keras_layers = [keras_layer]
    if wo_linearity:
        pass
        keras_layers += [Dense(2)]
    else:
        keras_layers += [Activation("relu"), Dense(2)]
    keras_model = Sequential(keras_layers)
    _ = keras_model(np.ones((1, input_dim)))

    return keras_model


def build_torch_model(keras_layer, torch_layer, keras_model, input_shape, input_dim, wo_linearity=False):
    class TorchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.torch_layer = torch_layer
            self.dense_0 = Linear(input_dim, np.prod(input_shape))
            self.inner_dim = np.prod(keras_layer.output.shape[1:])
            self.dense_1 = Linear(self.inner_dim, 2)

            if len(keras_layer.get_weights()):
                self.layers = [self.dense_0, self.torch_layer, self.dense_1]
            else:
                self.layers = [self.dense_0, self.dense_1]

            self.relu = nn.ReLU()

        def forward(self, x):
            y_0 = self.dense_0(x)
            y_1 = y_0.reshape([-1] + list(input_shape))
            y_2 = self.torch_layer(y_1)
            y_3 = y_2.reshape([-1, self.inner_dim])
            if wo_linearity:
                y_4 = y_3
            else:
                y_4 = self.relu(y_3)
            y_5 = self.dense_1(y_4)
            return y_5

    torch_model = TorchModel().to("cpu")
    return torch_model


def check_layer_linear(keras_layer, input_shape, method, decimal=6):
    input_dim = 30
    batch_size = 1

    keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=True)
    np_input = np.reshape(5 * np.random.rand(batch_size * input_dim) - 2, (batch_size, input_dim))
    torch_input = torch.Tensor(np_input)
    eps = 0.5
    decomon_model = clone(keras_model, final_ibp=True, final_affine=True, method=method)
    bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
    k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

    # compute bias
    bias = keras_model.predict(0 * np_input)[0]
    np.testing.assert_almost_equal(bias, k_ubias, decimal=decimal)
    np.testing.assert_almost_equal(bias, k_lbias, decimal=decimal)

    # compute weights
    weights = keras_model.predict(np.diag([1.0] * 30))
    np.testing.assert_almost_equal(weights, k_lA, decimal=decimal)
    np.testing.assert_almost_equal(weights, k_uA, decimal=decimal)


def share_weights_torch_2_keras(torch_model, keras_model, axis_to_permute_kernel=(2, 3, 1, 0)):
    # copy weights from torch to keras
    keras_params = []
    for layer in torch_model.layers:
        t_w, t_b = layer.state_dict().values()
        if len(t_w.shape) == 2:
            keras_params.append(t_w.T)
        elif len(t_w.shape) == 4:
            w = K.transpose(t_w, axis_to_permute_kernel)
            keras_params.append(w)
        else:
            # keras_params.append(K.transpose(t_w, (2, 1, 0)))
            keras_params.append(K.transpose(t_w, axis_to_permute_kernel))

        keras_params.append(t_b)

    keras_model.set_weights(keras_params)


def empirical_check_layer(keras_layer, input_shape, method, wo_linearity=False, decimal=6, keras_model=None):
    input_dim = 100
    batch_size = 1

    if keras_model is None:
        keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity)
    else:
        input_dim = keras_model.inputs[0].shape[-1]
        # keras_model = build_keras_model_maxpool(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity)
    # clip to positive values
    np_input = np.reshape(5 * np.random.rand(batch_size * input_dim) - 2, (batch_size, input_dim))
    eps = 0.01
    bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
    decomon_model = clone(keras_model, final_ibp=True, final_affine=True, method=method)
    k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

    if wo_linearity and method in ["forward-affine", "crown", "crown-forward-affine"]:
        # the affine bounds should be the same
        np.testing.assert_almost_equal(k_lA, k_uA)
        np.testing.assert_almost_equal(k_lbias, k_ubias)

    # sampling
    N = 100
    coeff = np.reshape(np.clip(np.random.rand(input_dim * N), 0, 1), (N, input_dim))
    sampling = (np_input - eps) * coeff + (1 - coeff) * (np_input + eps)
    pred_sampling = keras_model.predict(sampling)
    upper = np.sum(k_uA * sampling[:, :, None], 1) + k_ubias
    lower = np.sum(k_lA * sampling[:, :, None], 1) + k_lbias

    np.testing.assert_array_less(k_lb + 0 * upper - 10 ** (-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, k_ub + 0 * upper + 10 ** (-decimal))
    np.testing.assert_array_less(lower - 10 ** (-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, upper + 10 ** (-decimal))


def check_layer(
    keras_layer,
    torch_layer,
    input_shape,
    method,
    axis_to_permute_kernel=(2, 3, 1, 0),
    wo_linearity=False,
    decimal=3,
    mapping_keras2decomon_classes={},
    keras_model=None,
):
    input_dim = 30
    batch_size = 2

    if keras_model is None:
        keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity)
    else:
        input_dim = keras_model.inputs[0].shape[-1]
    torch_model = build_torch_model(
        keras_layer, torch_layer, keras_model, input_shape, input_dim, wo_linearity=wo_linearity
    )

    share_weights_torch_2_keras(torch_model, keras_model, axis_to_permute_kernel)

    # compare the output on the same random inputs
    np_input = np.reshape(5 * np.random.rand(batch_size * input_dim) - 2, (batch_size, input_dim))
    torch_input = torch.Tensor(np_input)
    output_torch = torch_model(torch_input)
    output_keras = keras_model.predict(np_input)
    np.testing.assert_almost_equal(output_keras, output_torch.detach().cpu().numpy(), decimal=decimal)

    auto_lirpa_model = BoundedModule(torch_model, torch_input)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.5)
    bounded_input = BoundedTensor(torch_input, ptb)

    auto_lirpa_method = map_decomon_methods_2_lirpa(method)
    default_bound_opts = {
        "sparse_intermediate_bounds": False,
        "sparse_intermediate_bounds_with_ibp": False,
    }
    auto_lirpa_model.set_bound_opts(default_bound_opts)

    if method in ["crown-forward-ibp", "crown"]:
        # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
        # Getting the linear bound coefficients (A matrix).
        required_A = defaultdict(set)
        required_A[auto_lirpa_model.output_name[0]].add(auto_lirpa_model.input_name[0])
        t_lb, t_ub, A = auto_lirpa_model.compute_bounds(
            x=(bounded_input,), method=auto_lirpa_method, return_A=True, needed_A_dict=required_A
        )
        # CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias
        t_lA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["lA"]
        t_lbias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["lbias"]
        t_uA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["uA"]
        t_ubias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["ubias"]

        eps = 0.5
        decomon_model = clone(keras_model, final_ibp=True, final_affine=True, method=method)

        bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
        k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

        if wo_linearity:
            k_lA = k_lA[None]
            k_uA = k_uA[None]
            k_lbias = k_lbias[None]
            k_ubias = k_ubias[None]
            t_lA = t_lA[:1]
            t_lbias = t_lbias[:1]
            t_uA = t_uA[:1]
            t_ubias = t_ubias[:1]
        k_lA = np.transpose(k_lA, (0, 2, 1))
        k_uA = np.transpose(k_uA, (0, 2, 1))

        np.testing.assert_almost_equal(k_lA, t_lA.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_uA, t_uA.detach().cpu().numpy(), decimal=decimal)

        np.testing.assert_almost_equal(k_lbias, t_lbias.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_ubias, t_ubias.detach().cpu().numpy(), decimal=decimal)
    else:
        # IBP only
        t_lb, t_ub = auto_lirpa_model.compute_bounds(x=(bounded_input,), method=auto_lirpa_method)
        eps = 0.5
        decomon_model = clone(
            keras_model,
            final_ibp=True,
            final_affine=False,
            method=method,
            mapping_keras2decomon_classes=mapping_keras2decomon_classes,
        )
        bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
        k_lb, k_ub = decomon_model.predict(bounds)
        decomon_model = clone(keras_model, final_ibp=True, final_affine=True, method=method)

    np.testing.assert_almost_equal(k_lb, t_lb.detach().cpu().numpy(), decimal=decimal)
    np.testing.assert_almost_equal(k_ub, t_ub.detach().cpu().numpy(), decimal=decimal)


def check_layer_activation(keras_layer, torch_layer, input_dim, method, decimal=6, finetune=False):
    batch_size = 2

    keras_model = Sequential([Dense(input_dim), keras_layer, Dense(input_dim)])
    _ = keras_model(np.ones((1, input_dim)))

    torch_model = nn.Sequential(Linear(input_dim, input_dim), torch_layer, Linear(input_dim, input_dim)).to("cpu")
    _ = torch_model(torch.ones((1, input_dim)))
    w_ = torch.tensor(np.asarray(np.diag([1] * input_dim), "float32"))
    b_ = torch.tensor(np.zeros(input_dim, dtype="float32"))
    keys = torch_model.state_dict().keys()
    dico_weights = {}
    weights = [w_, b_] * 2
    for i, key in enumerate(keys):
        dico_weights[key] = weights[i]
    torch_model.load_state_dict(dico_weights)

    keras_model.set_weights(weights)

    # compare the output on the same random inputs
    np_input = np.reshape(5 * np.random.rand(batch_size * input_dim) - 2, (batch_size, input_dim))
    torch_input = torch.Tensor(np_input)
    output_torch = torch_model(torch_input)
    output_keras = keras_model.predict(np_input)
    np.testing.assert_almost_equal(output_keras, output_torch.detach().cpu().numpy(), decimal=decimal)

    auto_lirpa_model = BoundedModule(torch_model, torch_input)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.5)
    bounded_input = BoundedTensor(torch_input, ptb)
    default_bound_opts = {
        "sparse_intermediate_bounds": False,
        "sparse_intermediate_bounds_with_ibp": False,
    }
    auto_lirpa_model.set_bound_opts(default_bound_opts)

    auto_lirpa_method = map_decomon_methods_2_lirpa(method)
    if method in ["crown-forward-ibp", "crown", "forward-affine", "forward-hybrid"]:
        # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
        # Getting the linear bound coefficients (A matrix).
        required_A = defaultdict(set)
        required_A[auto_lirpa_model.output_name[0]].add(auto_lirpa_model.input_name[0])
        t_lb, t_ub, A = auto_lirpa_model.compute_bounds(
            x=(bounded_input,), method=auto_lirpa_method, return_A=True, needed_A_dict=required_A
        )
        # CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias
        t_lA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["lA"]
        t_lbias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["lbias"]
        t_uA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["uA"]
        t_ubias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]["ubias"]

        eps = 0.5
        decomon_model = clone(keras_model, final_ibp=True, final_affine=True, method=method, finetune=finetune)
        bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
        k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

        # reshape weights of the affine bounds
        k_lA = np.transpose(k_lA, (0, 2, 1))
        k_uA = np.transpose(k_uA, (0, 2, 1))
        np.testing.assert_almost_equal(k_lA, t_lA.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_uA, t_uA.detach().cpu().numpy(), decimal=decimal)

        np.testing.assert_almost_equal(k_lbias, t_lbias.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_ubias, t_ubias.detach().cpu().numpy(), decimal=decimal)
    else:
        # IBP only
        t_lb, t_ub = auto_lirpa_model.compute_bounds(x=(bounded_input,), method=auto_lirpa_method)
        eps = 0.5
        decomon_model = clone(keras_model, final_ibp=True, final_affine=False, method=method)
        bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
        k_lb, k_ub = decomon_model.predict(bounds)

    np.testing.assert_almost_equal(k_lb, t_lb.detach().cpu().numpy(), decimal=decimal)
    np.testing.assert_almost_equal(k_ub, t_ub.detach().cpu().numpy(), decimal=decimal)


def empirical_check_layer_backward_non_linear(keras_layer, input_shape, method, wo_linearity=False, decimal=6):
    input_dim = 100
    batch_size = 1

    keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity, output_dim=1)

    # use jacobinet to create backward layer
    # Placeholder gradient to compute the Jacobian
    gradient_placeholder = keras.Variable(np.ones((1, 1), dtype="float32"))

    # Compute backward model for Jacobian calculation
    backward_model = clone_to_backward(keras_model, gradient=gradient_placeholder)
    # keras_model = build_keras_model_maxpool(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity)
    # clip to positive values
    np_input = np.reshape(5 * np.random.rand(batch_size * input_dim) - 2, (batch_size, input_dim))
    eps = 0.01
    bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)
    decomon_model = clone(backward_model, final_ibp=True, final_affine=True, method=method)
    k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

    if wo_linearity:
        # the affine bounds should be the same
        np.testing.assert_almost_equal(k_lA, k_uA)
        np.testing.assert_almost_equal(k_lbias, k_ubias)

    # sampling
    N = 100
    coeff = np.reshape(np.clip(np.random.rand(input_dim * N), 0, 1), (N, input_dim))
    sampling = (np_input - eps) * coeff + (1 - coeff) * (np_input + eps)
    pred_sampling = backward_model.predict(sampling)
    upper = np.sum(k_uA * sampling[:, :, None], 1) + k_ubias
    lower = np.sum(k_lA * sampling[:, :, None], 1) + k_lbias

    np.testing.assert_array_less(k_lb + 0 * upper - 10 ** (-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, k_ub + 0 * upper + 10 ** (-decimal))
    np.testing.assert_array_less(lower - 10 ** (-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, upper + 10 ** (-decimal))


def empirical_check_layer_backward_linear(keras_layer, input_shape, method, decimal=6):
    input_dim = 100
    batch_size = 1

    keras_model = build_keras_model(keras_layer, input_shape, input_dim, wo_linearity=True, output_dim=1)

    # use jacobinet to create backward layer
    # Placeholder gradient to compute the Jacobian

    # Compute backward model for Jacobian calculation
    backward_model = clone_to_backward(keras_model)
    # keras_model = build_keras_model_maxpool(keras_layer, input_shape, input_dim, wo_linearity=wo_linearity)
    # clip to positive values
    np_input = np.reshape(5 * np.random.rand(batch_size * 1) - 2, (batch_size, 1))
    eps = 0.01
    bounds = np.concatenate([np_input[:, None] - eps, np_input[:, None] + eps], 1)

    _ = backward_model.predict(np_input)
    decomon_model = clone(backward_model, final_ibp=True, final_affine=True, method=method)
    k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model.predict(bounds)

    np.testing.assert_almost_equal(k_lA, k_uA)
    np.testing.assert_almost_equal(k_lbias, k_ubias)

    """
    # sampling
    N = 100
    coeff = np.reshape(np.clip(np.random.rand(input_dim*N), 0, 1), (N, input_dim))
    sampling = (np_input-eps)*coeff + (1-coeff)*(np_input+eps)
    pred_sampling = backward_model.predict(sampling)
    upper = np.sum(k_uA*sampling[:,:,None], 1) + k_ubias
    lower = np.sum(k_lA*sampling[:,:,None], 1) + k_lbias

    np.testing.assert_array_less(k_lb+0*upper-10**(-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, k_ub+ 0*upper +10**(-decimal))
    np.testing.assert_array_less(lower-10**(-decimal), pred_sampling)
    np.testing.assert_array_less(pred_sampling, upper+10**(-decimal))"
    """
