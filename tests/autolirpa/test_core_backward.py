import pytest
from keras.layers import Dense

pytest.skip(
    reason="Decomon for backward layers not yet mature. "
    "Bounds for backward layers are different from autolirpa ones.",
    allow_module_level=True,
)


def _test_backward_dense_empirical_backward(units, input_shape, method, helpers):
    keras_layer = Dense(units)
    helpers.empirical_check_layer_backward_linear(keras_layer, input_shape, method=method, decimal=5)


# @pytest.mark.parametrize("method", ["forward-affine", "forward-hybrid", "crown-forward-ibp", "crown"])
@pytest.mark.parametrize("method", ["forward-ibp"])
def test_backward_dense_empirical_backward(method, helpers):
    input_shape = (10,)
    units = 100
    _test_backward_dense_empirical_backward(units, input_shape, method, helpers)
    # _test_backward_dense_empirical_backward(units, input_shape, method, wo_linearity=True)

    input_shape = (11,)
    units = 101
    # _test_backward_dense_empirical_backward(units, input_shape, method, wo_linearity=False)
    # _test_backward_dense_empirical_backward(units, input_shape, method, wo_linearity=True)
