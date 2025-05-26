import keras
import pytest
import torch
from keras.layers import (
    Cropping1D,
    Cropping2D,
    Cropping3D,
    Flatten,
    Permute,
    RepeatVector,
    Reshape,
    UpSampling1D,
    UpSampling2D,
    ZeroPadding1D,
    ZeroPadding2D,
    ZeroPadding3D,
)

from .conftest import check_layer, check_layer_linear, empirical_check_layer


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
            return x.view([-1] + self.target_shape)

    torch_layer = TorchReshape(target_shape)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=6)


def _test_backward_reshape_empirical(input_shape, target_shape, method):
    # data_format == 'channels_first'
    keras_layer = Reshape(target_shape)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=4)


def _test_backward_flatten(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = Flatten(data_format=data_format)
    if data_format == "channels_last":
        torch_layer = torch.nn.Flatten()
    else:

        class TorchFlatten(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.torch_layer = torch.nn.Flatten()

            def forward(self, x):
                """
                In the forward function we accept a Tensor of input data and we must return
                a Tensor of output data. We can use Modules defined in the constructor as
                well as arbitrary operators on Tensors.
                """
                x = torch.permute(x, [0, 2, 3, 1])
                return self.torch_layer(x)

        torch_layer = TorchFlatten()
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=6)


def _test_backward_flatten_empirical(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = Flatten(data_format=data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=4)


def _test_backward_permute(input_shape, axis, method):
    # data_format == 'channels_first'
    keras_layer = Permute(axis)

    class TorchPermute(torch.nn.Module):
        def __init__(self, axis) -> None:
            super().__init__()
            self.axis = [0] + list(axis)

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            return torch.permute(x, self.axis)

    torch_layer = TorchPermute(axis)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_permute_empirical(input_shape, axis, method):
    keras_layer = Permute(axis)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_cropping2d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = Cropping2D()
    torch_layer = keras_layer
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_cropping2d_empirical(input_shape, method, data_format):
    keras_layer = Cropping2D(data_format=data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=4)


def _test_backward_zeropadding2d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding2D(2)
    torch_layer = torch.nn.ZeroPad2d(2)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_zeropadding2d_empirical(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding2D(2, data_format=data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_upsampling2d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = UpSampling2D(2)
    torch_layer = torch.nn.Upsample(scale_factor=2)
    check_layer_linear(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_upsampling2d_empirical(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = UpSampling2D(2, data_format=data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_upsampling1d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = UpSampling1D(2)
    torch_layer = torch.nn.Upsample(scale_factor=2)
    check_layer_linear(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_upsampling1d_empirical(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = UpSampling1D(2)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_zeropadding1d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding1D(2)
    torch_layer = torch.nn.ZeroPad1d(2)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_zeropadding1d_empirical(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding1D(2, data_format=data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_zeropadding3d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding3D(2)
    torch_layer = torch.nn.ZeroPad3d(2)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=6)


def _test_backward_zeropadding3d_empirical(input_shape, method, data_format="channels_last"):
    # data_format == 'channels_first'
    keras_layer = ZeroPadding3D(2, data_format)
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=6)


def _test_backward_cropping1d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = Cropping1D()
    torch_layer = keras_layer
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_cropping1d_empirical(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = Cropping1D()
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_cropping3d(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = Cropping3D()
    torch_layer = keras_layer
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=5)


def _test_backward_cropping3d_empirical(input_shape, method):
    # data_format == 'channels_first'
    keras_layer = Cropping3D()
    empirical_check_layer(keras_layer, input_shape, method=method, decimal=5)


def _test_backward_repeatvector(input_shape, n, method):
    # data_format == 'channels_first'
    keras_layer = RepeatVector(n)

    class TorchRepeat(torch.nn.Module):
        def __init__(self, n) -> None:
            super().__init__()
            self.n = n

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            return torch.cat([x] * self.n, 1)
            # return x.repeat(1, self.n)

    torch_layer = TorchRepeat(n)
    check_layer(keras_layer, torch_layer, input_shape, method=method, decimal=0)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_reshape(method):
    keras.config.set_image_data_format("channels_first")

    input_shape = (2, 5, 10)
    target_shape = (2, 50)
    _test_backward_reshape(input_shape, target_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_reshape_empirical(method):
    keras.config.set_image_data_format("channels_first")

    input_shape = (2, 5, 10)
    target_shape = (2, 50)
    _test_backward_reshape_empirical(input_shape, target_shape, method)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_flatten(method):
    keras.config.set_image_data_format("channels_first")

    input_shape = (2, 5, 10)
    _test_backward_flatten(input_shape, method, data_format="channels_last")
    _test_backward_flatten(input_shape, method, data_format="channels_first")


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_flatten_empirical(method):
    keras.config.set_image_data_format("channels_first")

    input_shape = (2, 5, 10)
    _test_backward_flatten_empirical(input_shape, method, data_format="channels_last")
    _test_backward_flatten_empirical(input_shape, method, data_format="channels_first")


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_RepeatVector(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (10,)
    _test_backward_repeatvector(input_shape, 2, method)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Permute(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 5, 10)
    axis = (3, 2, 1)
    _test_backward_permute(input_shape, axis, method)

    input_shape = (2, 4, 10)
    axis = (3, 1, 2)
    _test_backward_permute(input_shape, axis, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_Permute_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 5, 10)
    axis = (3, 2, 1)
    _test_backward_permute_empirical(input_shape, axis, method)

    input_shape = (2, 4, 10)
    axis = (3, 1, 2)
    _test_backward_permute_empirical(input_shape, axis, method)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Cropping2D(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10)
    _test_backward_cropping2d(input_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_Cropping2D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10)
    _test_backward_cropping2d_empirical(input_shape, method, data_format="channels_first")
    input_shape = (10, 10, 2)
    _test_backward_cropping2d_empirical(input_shape, method, data_format="channels_last")


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Cropping1D(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (3, 2)
    _test_backward_cropping1d(input_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_Cropping1D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (3, 2)
    _test_backward_cropping1d_empirical(input_shape, method)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_Cropping3D(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10, 10)
    _test_backward_cropping3d(input_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_Cropping3D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10, 10)
    _test_backward_cropping3d_empirical(input_shape, method)


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding2D(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10)
    _test_backward_zeropadding2d(input_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding2D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10)
    _test_backward_zeropadding2d_empirical(input_shape, method, data_format="channels_first")
    input_shape = (10, 10, 2)
    _test_backward_zeropadding2d_empirical(input_shape, method, data_format="channels_last")


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding1D(method):
    # not implemented in auto lirpa
    pass


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding1D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    # not implemented in auto lirpa
    input_shape = (2, 10)
    _test_backward_zeropadding1d_empirical(input_shape, method, data_format="channels_last")
    input_shape = (10, 2)
    _test_backward_zeropadding1d_empirical(input_shape, method, data_format="channels_first")


@pytest.mark.parametrize("method", ["forward-ibp", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding3D(method):
    # not implemented in auto lirpa
    pass


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_ZeroPadding3D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    # not implemented in auto lirpa
    input_shape = (2, 10, 10, 10)
    _test_backward_zeropadding3d_empirical(input_shape, method, data_format="channels_first")
    input_shape = (10, 10, 10, 2)
    _test_backward_zeropadding3d_empirical(input_shape, method, data_format="channels_last")


@pytest.mark.parametrize("method", ["crown"])
def test_backward_UpSampling2D(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10, 10)
    _test_backward_upsampling2d(input_shape, method)


@pytest.mark.parametrize(
    "method, data_format",
    [
        ("forward-affine", "channels_first"),
        ("forward-hybrid", "channels_first"),
        ("crown-forward-ibp", "channels_first"),
        ("crown", "channels_first"),
        ("forward-affine", "channels_last"),
        ("forward-hybrid", "channels_last"),
        ("crown-forward-ibp", "channels_last"),
        ("crown", "channels_last"),
    ],
)
def test_backward_UpSampling2D_empirical(method, data_format):
    keras.config.set_image_data_format(data_format)
    if data_format == "channels_first":
        input_shape = (2, 10, 10)
    else:
        input_shape = (10, 10, 2)
    _test_backward_upsampling2d_empirical(input_shape, method, data_format=data_format)


@pytest.mark.parametrize("method", ["crown"])
def test_backward_UpSampling1D(method):
    pass
    input_shape = (2, 10)
    _test_backward_upsampling1d(input_shape, method)


@pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
def test_backward_UpSampling1D_empirical(method):
    keras.config.set_image_data_format("channels_first")
    input_shape = (2, 10)
    _test_backward_upsampling1d_empirical(input_shape, method)
