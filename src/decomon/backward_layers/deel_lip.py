import logging
from typing import Any, Dict, Optional, Union

from keras.layers import Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.core import ForwardMode, PerturbationDomain, get_affine, get_ibp
from decomon.layers.convert import to_decomon
from decomon.layers.decomon_layers import DecomonLayer

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
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )

        if not isinstance(layer, DecomonLayer):
            self.layer = to_decomon(
                layer=layer,
                input_dim=input_dim,
                dc_decomp=False,
                perturbation_domain=self.perturbation_domain,
                finetune=False,
                ibp=get_ibp(self.mode),
                affine=get_affine(self.mode),
                shared=True,
                fast=False,
            )

        self.frozen_weights = False
