import logging
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import get_affine, get_ibp
from decomon.layers.core import ForwardMode
from decomon.layers.decomon_layers import DecomonLayer, to_decomon
from decomon.utils import Slope

logger = logging.getLogger(__name__)

try:
    from deel.lip.activations import GroupSort, GroupSort2
except ImportError:
    logger.warning(
        "Could not import GroupSort or GroupSort2 from deel.lip.activations. "
        "Please install deel-lip for being compatible with 1 Lipschitz network (see https://github.com/deel-ai/deel-lip)"
    )


class BackwardGroupSort2(BackwardLayer):
    """Backward LiRPA of GroupSort2"""

    def __init__(
        self,
        layer: Layer,
        input_dim: int = -1,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

        if not isinstance(layer, DecomonLayer):
            self.layer = to_decomon(
                layer,
                input_dim,
                dc_decomp=False,
                convex_domain=self.convex_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )

        self.frozen_weights = False
