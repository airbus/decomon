# creating toy network and assess that the decomposition is correct


import numpy as np
import pytest
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Sequential

from decomon.backward_layers.utils import get_affine, get_ibp
from decomon.layers.core import ForwardMode, get_mode
from decomon.layers.decomon_reshape import DecomonReshape
from decomon.models import clone
from decomon.models.utils import ConvertMethod, get_ibp_affine_from_method
from decomon.utils import Slope

try:
    import deel.lip
except ImportError:
    deel_lip_available = False
else:
    deel_lip_available = True
    from deel.lip.activations import GroupSort
    from deel.lip.layers import (
        FrobeniusDense,
        ScaledL2NormPooling2D,
        SpectralConv2D,
        SpectralDense,
    )
    from deel.lip.model import Sequential as DeellipSequential


deel_lip_skip_reason = "deel-lip is not available"


def test_convert_1D(n, method, mode, floatx, decimal, helpers):
    if not helpers.is_method_mode_compatible(method=method, mode=mode):
        # skip method=ibp/crown-ibp with mode=affine/hybrid
        pytest.skip(f"output mode {mode} is not compatible with convert method {method}")

    dc_decomp = False
    ibp = get_ibp(mode=mode)
    affine = get_affine(mode=mode)

    # numpy inputs
    inputs_ = helpers.get_standard_values_1d_box(n, dc_decomp=dc_decomp)
    input_ref_ = helpers.get_input_ref_from_full_inputs(inputs_)
    input_decomon_ = helpers.get_inputs_np_for_decomon_model_from_full_inputs(inputs=inputs_)

    #  keras model and output of reference
    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    output_ref_ = ref_nn.predict(input_ref_)

    # decomon conversion
    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine)

    #  decomon outputs
    outputs_ = decomon_model.predict(input_decomon_)

    #  check bounds consistency
    helpers.assert_decomon_model_output_properties_box(
        full_inputs=inputs_,
        output_ref=output_ref_,
        outputs_for_mode=outputs_,
        mode=mode,
        dc_decomp=dc_decomp,
        decimal=decimal,
    )


def test_convert_1D_forward_slope(slope, helpers):
    ibp = True
    affine = True
    dc_decomp = False

    n, method, mode = 0, "forward-hybrid", "hybrid"
    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(input_ref)

    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of activation layers
    for layer in decomon_model.layers:
        if layer.__class__.__name__.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_convert_1D_backward_slope(slope, helpers):
    n, method, mode = 0, "crown-forward-hybrid", "hybrid"
    ibp = True
    affine = True
    dc_decomp = False

    inputs = helpers.get_tensor_decomposition_1d_box(dc_decomp=dc_decomp)
    input_ref = helpers.get_input_ref_from_full_inputs(inputs=inputs)

    ref_nn = helpers.toy_network_tutorial(dtype=K.floatx())
    ref_nn(input_ref)

    decomon_model = clone(ref_nn, method=method, final_ibp=ibp, final_affine=affine, slope=slope)

    # check slope of layers with activation
    for layer in decomon_model.layers:
        layer_class_name = layer.__class__.__name__
        if layer_class_name.endswith("Activation"):
            assert layer.slope == Slope(slope)


def test_name_forward():
    layers = []
    layers.append(Dense(1, input_dim=1))
    layers.append(Dense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(Dense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_f = clone(model=model, method=ConvertMethod.FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 1
    nb_batman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 2


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
def test_name_forward_deellip():
    layers = []
    layers.append(Dense(1, input_dim=1))
    layers.append(SpectralDense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(SpectralDense(1, activation=GroupSort(n=1), name="batman"))
    model = Sequential(layers)

    decomon_model_f = clone(model=model, method=ConvertMethod.FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 1
    nb_batman_layers = len([layer for layer in decomon_model_f.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 2


def test_name_backward():
    layers = []
    layers.append(Dense(1, input_dim=1))
    layers.append(Dense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(Dense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_b = clone(model=model, method=ConvertMethod.CROWN_FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 2
    nb_batman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 3


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
def test_name_backward_deellip():
    layers = []
    layers.append(Dense(1, input_dim=1))
    layers.append(SpectralDense(1, name="superman"))  # specify the dimension of the input space
    layers.append(Activation("relu"))
    layers.append(SpectralDense(1, activation="relu", name="batman"))
    model = Sequential(layers)

    decomon_model_b = clone(model=model, method=ConvertMethod.CROWN_FORWARD_HYBRID)
    nb_superman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("superman_")])
    assert nb_superman_layers == 2
    nb_batman_layers = len([layer for layer in decomon_model_b.layers if layer.name.startswith("batman_")])
    assert nb_batman_layers == 3


@pytest.mark.skipif(not (deel_lip_available), reason=deel_lip_skip_reason)
def test_clone_full_deellip_model_forward(helpers):
    target_shape = (6, 6, 2)
    input_shape = (np.prod(target_shape),)
    model = DeellipSequential(
        [
            Reshape(target_shape=target_shape, input_shape=input_shape),
            # Lipschitz layers preserve the API of their superclass ( here Conv2D )
            # an optional param is available: k_coef_lip which control the lipschitz
            # constant of the layer
            SpectralConv2D(
                filters=16,
                kernel_size=(3, 3),
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            # our layers are fully interoperable with existing keras layers
            Flatten(),
            SpectralDense(
                32,
                activation=GroupSort(2),
                use_bias=True,
                kernel_initializer="orthogonal",
            ),
            FrobeniusDense(10, activation=None, use_bias=False, kernel_initializer="orthogonal"),
        ],
        # similary model has a parameter to set the lipschitz constant
        # to set automatically the constant of each layer
        k_coef_lip=1.0,
        name="hkr_model",
    )

    method = ConvertMethod.FORWARD_HYBRID
    ibp, affine = get_ibp_affine_from_method(method)
    mode = get_mode(ibp=ibp, affine=affine)
    decomon_model = clone(model, method=method, final_ibp=ibp, final_affine=affine)

    # check bounds: todo
