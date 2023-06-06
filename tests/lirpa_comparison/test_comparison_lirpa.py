# Goal: compute bounds with lirpa and decomon and assess that they are the same (up to numerical precision)

import numpy as np
import onnx
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from numpy.testing import assert_almost_equal
from onnx2keras import onnx_to_keras
from onnx2torch import convert

from decomon.models.convert import clone
from decomon.models.utils import ConvertMethod


@pytest.mark.parametrize(
    "path, onnx_filename, method, eps, N",
    [
        (".", "mnist_relu_3_50_comparison.onnx", "IBP", 0.01, 3),
        (".", "mnist_relu_3_50_comparison.onnx", "IBP", 0.1, 3),
        (".", "mnist_relu_3_50_comparison.onnx", "IBP+backward", 0.01, 3),
        (".", "mnist_relu_3_50_comparison.onnx", "IBP+backward", 0.1, 3),
        (".", "mnist_relu_3_50_comparison.onnx", "backward", 0.01, 3),
        (".", "mnist_relu_3_50_comparison.onnx", "backward", 0.1, 3),
    ],
)
def test_with_lirpa(path, onnx_filename, method, eps, N):
    decimal = 4
    equ_method = {
        "IBP": ConvertMethod.FORWARD_IBP,
        "backward": ConvertMethod.CROWN,
        "IBP+backward": ConvertMethod.CROWN_FORWARD_IBP,
    }

    # use use onnx2torch to load the network in torch
    onnx_model = onnx.load(onnx_filename)
    torch_model = clone(onnx_model)

    ## Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=torchvision.transforms.ToTensor()
    )
    N = 3
    # we consider the first N test examples correctly classified by the network
    image = test_data.data[: 2 * N].view(2 * N, 784) / 255.0 + 0.0
    # apply lirpa bounds
    norm = np.inf

    ptb = PerturbationLpNorm(norm=norm, eps=eps)
    image_tensor = BoundedTensor(image, ptb)
    lirpa_model = BoundedModule(torch_model, image, device="cpu")

    lb_, ub_ = lirpa_model.compute_bounds(x=(image_tensor,), method=method)
    lb_p = lb_.detach().numpy()
    ub_p = ub_.detach().numpy()

    onnx_model = onnx.load(onnx_filename)
    k_model = onnx_to_keras(onnx_model, ["0"])

    decomon_model = clone(k_model, method=equ_method[method], final_ibp=True, final_affine=False)

    image_ = image.detach().numpy()
    box = np.concatenate([image_[:, None] - eps, image_[:, None] + eps], 1)
    ub_k, lb_k = decomon_model.predict(box)

    assert_almost_equal(ub_k, ub_p, decimal=decimal, err_msg="error")
    assert_almost_equal(lb_k, lb_p, decimal=decimal, err_msg="error")
