import logging
from typing import Any, Optional

from keras.layers import Activation, Add, Dense, Layer

import decomon.layers
from decomon.core import PerturbationDomain, Propagation, Slope
from decomon.layers import DecomonActivation, DecomonAdd, DecomonDense, DecomonLayer

logger = logging.getLogger(__name__)


DECOMON_PREFIX = "Decomon"

default_mapping_keras2decomon_classes: dict[type[Layer], type[DecomonLayer]] = {
    Add: DecomonAdd,
    Dense: DecomonDense,
    Activation: DecomonActivation,
}
"""Default mapping between keras layers and decomon layers."""

default_mapping_kerasname2decomonclass: dict[str, type[DecomonLayer]] = {
    k[len(DECOMON_PREFIX) :]: v
    for k, v in vars(decomon.layers).items()
    if k.startswith(DECOMON_PREFIX) and issubclass(v, DecomonLayer)
}
"""Default mapping from a keras class name to a decomon layer class.

This mapping is generated automatically from `decomon.layers` namespace.
It is used only when `default_mapping_keras2decomon_classes` does not contain
the desired keras layer class.

"""


def to_decomon(
    layer: Layer,
    perturbation_domain: Optional[PerturbationDomain] = None,
    ibp: bool = True,
    affine: bool = True,
    propagation: Propagation = Propagation.FORWARD,
    model_input_shape: Optional[tuple[int, ...]] = None,
    model_output_shape: Optional[tuple[int, ...]] = None,
    slope: Slope = Slope.V_SLOPE,
    mapping_keras2decomon_classes: Optional[dict[type[Layer], type[DecomonLayer]]] = None,
    **kwargs: Any,
) -> DecomonLayer:
    """Convert a keras layer into the corresponding decomon layer.

    Args:
        layer: keras layer to convert
        perturbation_domain: perturbation domain. Default to a box domain
        ibp: if True, forward propagate constant bounds
        affine: if True, forward propagate affine bounds
        propagation: direction of bounds propagation
          - forward: from input to output
          - backward: from output to input
        model_output_shape: shape of the underlying model output (omitting batch axis).
           It allows determining if the backward bounds are with a batch axis or not.
        model_input_shape: shape of the underlying keras model input (omitting batch axis).
        slope: slope used for the affine relaxation of activation layers
        mapping_keras2decomon_classes: user-defined mapping between keras and decomon layers classes
        **kwargs: keyword-args passed to decomon layers

    Returns:
        the converted decomon layer

    The decomon class is chosen as followed (from higher to lower priority):
    - user-defined `mapping_keras2decomon_classes`,
    - default mapping `default_mapping_keras2decomon_classes`,
    - using keras class name, by adding a "Decomon" prefix, thanks to `default_mapping_kerasname2decomonclass`.

    """
    # Choose the corresponding decomon class. User mapping -> default mapping -> name.
    keras_class = type(layer)
    decomon_class = None
    if mapping_keras2decomon_classes is not None:
        decomon_class = mapping_keras2decomon_classes.get(keras_class, None)
    if decomon_class is None:
        decomon_class = default_mapping_keras2decomon_classes.get(keras_class, None)
    if decomon_class is None:
        logger.warning(
            f"Keras layer {layer} not in user-defined nor default mapping: "
            f"using class name to deduce the proper decomon class to use."
        )
        decomon_class = default_mapping_kerasname2decomonclass.get(keras_class.__name__, None)
    if decomon_class is None:
        raise NotImplementedError(f"The decomon version of {keras_class} is not yet implemented.")

    return decomon_class(
        layer=layer,
        perturbation_domain=perturbation_domain,
        ibp=ibp,
        affine=affine,
        propagation=propagation,
        model_input_shape=model_input_shape,
        model_output_shape=model_output_shape,
        slope=slope,
        **kwargs,
    )
