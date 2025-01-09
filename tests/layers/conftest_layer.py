# create a general test to compare a layer with auto lirpa

import numpy as np
import keras
import keras.ops as K
from keras.layers import Reshape, Dense, Activation
from keras.models import Sequential
import torch
from torch import nn
from torch.nn import Linear
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from collections import defaultdict

from decomon.models.convert import clone

def map_decomon_methods_2_lirpa(method:str):
    if method.lower()=='crown': return 'CROWN'
    elif method.lower()=='forward-ibp': return 'IBP'
    elif method.lower()=='crown-forward-ibp': return 'CROWN-IBP'
    else: raise ValueError('unrecognized method {}'.format(method))

def build_keras_model(keras_layer, input_shape, input_dim):
    keras_layers = [Dense(np.prod(input_shape)), Reshape(input_shape), keras_layer, Reshape((-1,)), Activation('relu'), Dense(2)]
    keras_model = Sequential(keras_layers)
    _ = keras_model(np.ones((1, input_dim)))

    return keras_model

def build_torch_model(keras_layer, torch_layer, keras_model, input_shape, input_dim):

    class TorchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
    
            self.torch_layer = torch_layer
            self.dense_0 = Linear(input_dim, np.prod(input_shape))
            self.inner_dim = keras_model.layers[-1].input.shape[-1]
            self.dense_1 = Linear(self.inner_dim, 2)
    
    
            if len(keras_layer.get_weights()):
                self.layers = [self.dense_0, self.torch_layer, self.dense_1]
            else:
                self.layers = [self.dense_0, self.dense_1]
            
            self.relu = nn.ReLU()

        def forward(self, x):
            y_0 = self.dense_0(x)
            y_1 = y_0.reshape([-1]+list(input_shape))
            y_2 = self.torch_layer(y_1)
            y_3 = y_2.reshape([-1, self.inner_dim])
            y_4 = self.relu(y_3)
            y_5 = self.dense_1(y_4)
            return y_5

    torch_model = TorchModel().to('cpu')
    return torch_model

    
def check_layer(keras_layer, torch_layer, input_shape, method, decimal=6):

    input_dim = 30
    batch_size = 2
    
    keras_model = build_keras_model(keras_layer, input_shape, input_dim)
    torch_model = build_torch_model(keras_layer, torch_layer, keras_model, input_shape, input_dim)

    # copy weights from torch to keras
    keras_params=[]
    for layer in torch_model.layers:
        t_w, t_b = layer.state_dict().values()
        if len(t_w.shape)==2:
            keras_params.append(t_w.T)
        else:
            keras_params.append(K.transpose(t_w, (2, 3, 1, 0)))
            
        keras_params.append(t_b)
    
    keras_model.set_weights(keras_params)
    
    # compare the output on the same random inputs
    np_input = np.reshape(5*np.random.rand(batch_size*input_dim)-2, (batch_size, input_dim))
    torch_input = torch.Tensor(np_input)
    output_torch = torch_model(torch_input)
    output_keras = keras_model(torch_input)
    np.testing.assert_almost_equal(output_keras.detach().cpu().numpy(), output_torch.detach().cpu().numpy(), decimal=decimal)

    auto_lirpa_model = BoundedModule(torch_model, torch_input)
    ptb = PerturbationLpNorm(norm=np.inf, eps=0.5)
    bounded_input = BoundedTensor(torch_input, ptb)
    
    auto_lirpa_method = map_decomon_methods_2_lirpa(method)
    if method in ['crown-forward-ibp', 'crown']:

        # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
        # Getting the linear bound coefficients (A matrix).    
        required_A = defaultdict(set)
        required_A[auto_lirpa_model.output_name[0]].add(auto_lirpa_model.input_name[0])
        t_lb, t_ub, A = auto_lirpa_model.compute_bounds(x=(bounded_input,), method=auto_lirpa_method, return_A=True, needed_A_dict=required_A)
        # CROWN linear (symbolic) bounds: lA x + lbias <= f(x) <= uA x + ubias
        t_lA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]['lA']
        t_lbias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]['lbias']
        t_uA = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]['uA']
        t_ubias = A[auto_lirpa_model.output_name[0]][auto_lirpa_model.input_name[0]]['ubias']

        eps = 0.5
        decomon_model = clone(keras_model,final_ibp=True, final_affine=True, method=method)
        bounds = K.concatenate([torch_input[:,None]-eps, torch_input[:,None]+eps], 1)
        k_lA, k_lbias, k_uA, k_ubias, k_lb, k_ub = decomon_model(bounds)
        
        # reshape weights of the affine bounds
        k_lA = K.transpose(k_lA, (0, 2, 1))
        k_uA = K.transpose(k_uA, (0, 2, 1))

        np.testing.assert_almost_equal(k_lA.detach().cpu().numpy(), t_lA.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_uA.detach().cpu().numpy(), t_uA.detach().cpu().numpy(), decimal=decimal)
    
        np.testing.assert_almost_equal(k_lbias.detach().cpu().numpy(), t_lbias.detach().cpu().numpy(), decimal=decimal)
        np.testing.assert_almost_equal(k_ubias.detach().cpu().numpy(), t_ubias.detach().cpu().numpy(), decimal=decimal)
    else:
        # IBP only
        t_lb, t_ub = auto_lirpa_model.compute_bounds(x=(bounded_input,), method=auto_lirpa_method)
        eps = 0.5
        decomon_model = clone(keras_model,final_ibp=True, final_affine=False, method=method)
        bounds = K.concatenate([torch_input[:,None]-eps, torch_input[:,None]+eps], 1)
        k_lb, k_ub = decomon_model(bounds)

    np.testing.assert_almost_equal(k_lb.detach().cpu().numpy(), t_lb.detach().cpu().numpy(), decimal=decimal)
    np.testing.assert_almost_equal(k_ub.detach().cpu().numpy(), t_ub.detach().cpu().numpy(), decimal=decimal)
    
    