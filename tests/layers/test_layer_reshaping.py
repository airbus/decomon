import pytest

from keras.layers import (
    Reshape,
    Flatten,
    Cropping2D,
    ZeroPadding2D,
    Cropping1D,
    ZeroPadding1D,
    Permute,
    RepeatVector,
    Cropping3D,
)
import torch

from .conftest_layer import check_layer


def _test_backward_reshape(input_shape, target_shape, method):

    # data_format == 'channels_first'
    keras_layer = Reshape(target_shape)

    class TorchReshape(torch.nn.Module):

        def __init__(self, target_shape) -> None:
            super().__init__()
            self.target_shape = list(target_shape)

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            return x.view([-1]+self.target_shape)
    

    torch_layer = TorchReshape(target_shape)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=6)

@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_reshape(method):

    input_shape = (2, 5, 10)
    target_shape = (2, 50)
    _test_backward_reshape(input_shape, target_shape, method)


def test_backward_RepeatVector():
    pass


def test_backward_Permute():
    pass

def test_backward_Flatten():
    pass

def test_backward_Cropping2D():
    pass

def test_backward_ZeroPadding2D():
    pass

def test_backward_Cropping1D():
    pass

def test_backward_ZeroPadding1D():
    pass

def test_backward_Cropping3D():
    pass


