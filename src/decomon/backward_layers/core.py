from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer, Wrapper

from decomon.core import BoxDomain, ForwardMode, PerturbationDomain
from decomon.layers.core import DecomonLayer


class BackwardLayer(ABC, Wrapper):

    layer: Layer
    _trainable_weights: List[tf.Variable]

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
    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        """
        Args:
            inputs

        Returns:

        """
        pass

    def build(self, input_shape: List[tf.TensorShape]) -> None:
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
