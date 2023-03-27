from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer, Wrapper

from decomon.layers.core import DecomonLayer, ForwardMode


class BackwardLayer(ABC, Wrapper):

    layer: Layer
    _trainable_weights: List[tf.Variable]

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        previous: bool = True,
        **kwargs: Any,
    ):
        kwargs.pop("slope", None)
        kwargs.pop("finetune", None)
        super().__init__(layer, **kwargs)
        self.rec = rec
        self.previous = previous
        if isinstance(self.layer, DecomonLayer):
            self.mode = self.layer.mode
            self.convex_domain = self.layer.convex_domain
            self.dc_decomp = self.layer.dc_decomp
        else:
            self.mode = ForwardMode(mode)
            if convex_domain is None:
                self.convex_domain = {}
            else:
                self.convex_domain = convex_domain
            self.dc_decomp = dc_decomp

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "rec": self.rec,
                "mode": self.mode,
                "convex_domain": self.convex_domain,
                "dc_decomp": self.dc_decomp,
                "previous": self.previous,
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
