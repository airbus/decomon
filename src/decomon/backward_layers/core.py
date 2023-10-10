from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import keras_core as keras
from keras_core.layers import Layer, Wrapper

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
)
from decomon.layers.core import DecomonLayer


class BackwardLayer(ABC, Wrapper):
    layer: Layer
    _trainable_weights: List[keras.Variable]

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        kwargs.pop("slope", None)
        kwargs.pop("finetune", None)
        super().__init__(layer, **kwargs)
        self.rec = rec
        if isinstance(self.layer, DecomonLayer):
            self.mode = self.layer.mode
            self.perturbation_domain = self.layer.perturbation_domain
            self.dc_decomp = self.layer.dc_decomp
        else:
            self.mode = ForwardMode(mode)
            if perturbation_domain is None:
                self.perturbation_domain = BoxDomain()
            else:
                self.perturbation_domain = perturbation_domain
            self.dc_decomp = dc_decomp
        self.inputs_outputs_spec = InputsOutputsSpec(
            dc_decomp=self.dc_decomp, mode=self.mode, perturbation_domain=self.perturbation_domain
        )

    @property
    def ibp(self) -> bool:
        return get_ibp(self.mode)

    @property
    def affine(self) -> bool:
        return get_affine(self.mode)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "rec": self.rec,
                "mode": self.mode,
                "perturbation_domain": self.perturbation_domain,
                "dc_decomp": self.dc_decomp,
            }
        )
        return config

    @abstractmethod
    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        """
        Args:
            inputs

        Returns:

        """
        pass

    def build(self, input_shape: List[Tuple[Optional[int]]]) -> None:
        """
        Args:
            input_shape

        Returns:

        """
        # generic case: nothing to do before call
        pass

    def freeze_weights(self) -> None:
        pass

    def unfreeze_weights(self) -> None:
        pass

    def freeze_alpha(self) -> None:
        pass

    def unfreeze_alpha(self) -> None:
        pass

    def reset_finetuning(self) -> None:
        pass
