from typing import Any, Dict, Optional, Union

from tensorflow.keras.layers import Layer

import decomon.backward_layers.backward_layers
import decomon.backward_layers.backward_maxpooling
import decomon.backward_layers.backward_merge
import decomon.backward_layers.deel_lip
from decomon.backward_layers.core import BackwardLayer
from decomon.layers.core import ForwardMode
from decomon.utils import Slope

_mapping_name2class: Dict[str, Any] = vars(decomon.backward_layers.backward_layers)
_mapping_name2class.update(vars(decomon.backward_layers.deel_lip))
_mapping_name2class.update(vars(decomon.backward_layers.backward_merge))
_mapping_name2class.update(vars(decomon.backward_layers.backward_maxpooling))


def to_backward(
    layer: Layer,
    slope: Union[str, Slope] = Slope.V_SLOPE,
    mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    **kwargs: Any,
) -> BackwardLayer:
    if convex_domain is None:
        convex_domain = {}
    class_name = layer.__class__.__name__
    if class_name.startswith("Decomon"):
        class_name = "".join(layer.__class__.__name__.split("Decomon")[1:])

    backward_class_name = f"Backward{class_name}"
    try:
        class_ = _mapping_name2class[backward_class_name]
    except KeyError:
        raise NotImplementedError(f"The backward version of {class_name} is not yet implemented.")
    backward_layer_name = f"{layer.name}_backward"
    return class_(
        layer,
        slope=slope,
        mode=mode,
        convex_domain=convex_domain,
        finetune=finetune,
        dtype=layer.dtype,
        name=backward_layer_name,
        **kwargs,
    )
