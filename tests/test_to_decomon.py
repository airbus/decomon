import logging
from importlib import reload
from typing import Any, Optional

import pytest
from keras.layers import Activation, Dense, Input, Layer

import decomon.layers
import decomon.layers.convert
from decomon.constants import Propagation, Slope
from decomon.layers import DecomonActivation, DecomonDense, DecomonLayer
from decomon.layers.convert import to_decomon
from decomon.perturbation_domain import BoxDomain, PerturbationDomain

# Add a new class in decomon.layers namespace for testing conversion by name
# We must do it *before* importing decomon.layers.convert
decomon.layers.DecomonToto = DecomonDense
reload(decomon.layers.convert)
from decomon.layers.convert import to_decomon


class Toto(Dense):
    ...


class MyBoxDomain(BoxDomain):
    ...


class MyDenseDecomonLayer(DecomonLayer):
    def __init__(
        self,
        layer: Layer,
        perturbation_domain: Optional[PerturbationDomain] = None,
        ibp: bool = True,
        affine: bool = True,
        propagation: Propagation = Propagation.FORWARD,
        model_input_shape: Optional[tuple[int, ...]] = None,
        model_output_shape: Optional[tuple[int, ...]] = None,
        my_super_attribute: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(
            layer, perturbation_domain, ibp, affine, propagation, model_input_shape, model_output_shape, **kwargs
        )
        self.my_super_attribute = my_super_attribute


class MyKerasDenseLayer(Dense):
    ...


ibp = True
affine = False
propagation = Propagation.BACKWARD
perturbation_domain = MyBoxDomain()
slope = Slope.Z_SLOPE
model_input_shape = (1,)
model_output_shape = (1,)


def test_to_decomon_userdefined():
    mymapping = {Dense: MyDenseDecomonLayer}
    my_super_attribute = 123.0
    layer = Dense(3)
    layer(Input((1,)))
    decomon_layer = to_decomon(
        layer=layer,
        perturbation_domain=perturbation_domain,
        ib=ibp,
        affine=affine,
        propagation=propagation,
        model_input_shape=model_input_shape,
        model_output_shape=model_output_shape,
        slope=slope,
        my_super_attribute=my_super_attribute,
        mapping_keras2decomon_classes=mymapping,
    )
    assert isinstance(decomon_layer, MyDenseDecomonLayer)

    assert decomon_layer.ibp == ibp
    assert decomon_layer.affine == affine
    assert decomon_layer.propagation == propagation
    assert decomon_layer.perturbation_domain == perturbation_domain
    assert decomon_layer.model_input_shape == model_input_shape
    assert decomon_layer.model_output_shape == model_output_shape

    assert decomon_layer.my_super_attribute == my_super_attribute


def test_to_decomon_default(caplog):
    layer = Dense(3)
    layer(Input((1,)))

    with caplog.at_level(logging.WARNING):
        decomon_layer = to_decomon(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ib=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            slope=slope,
        )

    assert isinstance(decomon_layer, DecomonDense)

    assert decomon_layer.ibp == ibp
    assert decomon_layer.affine == affine
    assert decomon_layer.propagation == propagation
    assert decomon_layer.perturbation_domain == perturbation_domain
    assert decomon_layer.model_input_shape == model_input_shape
    assert decomon_layer.model_output_shape == model_output_shape

    assert "using class name" not in caplog.text


def test_to_decomon_slope():
    layer = Activation(activation=None)
    layer(Input((1,)))
    decomon_layer = to_decomon(
        layer=layer,
        perturbation_domain=perturbation_domain,
        ib=ibp,
        affine=affine,
        propagation=propagation,
        model_input_shape=model_input_shape,
        model_output_shape=model_output_shape,
        slope=slope,
    )
    assert isinstance(decomon_layer, DecomonActivation)

    assert decomon_layer.ibp == ibp
    assert decomon_layer.affine == affine
    assert decomon_layer.propagation == propagation
    assert decomon_layer.perturbation_domain == perturbation_domain
    assert decomon_layer.model_input_shape == model_input_shape
    assert decomon_layer.model_output_shape == model_output_shape

    assert decomon_layer.slope == slope


def test_to_decomon_by_name(caplog):
    layer = Toto(3)
    layer(Input((1,)))

    with caplog.at_level(logging.WARNING):
        decomon_layer = to_decomon(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ib=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            slope=slope,
        )

    assert isinstance(decomon_layer, DecomonDense)

    assert decomon_layer.ibp == ibp
    assert decomon_layer.affine == affine
    assert decomon_layer.propagation == propagation
    assert decomon_layer.perturbation_domain == perturbation_domain
    assert decomon_layer.model_input_shape == model_input_shape
    assert decomon_layer.model_output_shape == model_output_shape

    assert "using class name" in caplog.text


def test_to_decomon_nok():
    layer = MyKerasDenseLayer(3)
    layer(Input((1,)))
    with pytest.raises(NotImplementedError):
        to_decomon(
            layer=layer,
            perturbation_domain=perturbation_domain,
            ib=ibp,
            affine=affine,
            propagation=propagation,
            model_input_shape=model_input_shape,
            model_output_shape=model_output_shape,
            slope=slope,
        )
