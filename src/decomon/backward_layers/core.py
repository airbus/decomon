from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
import numpy as np
from keras.layers import InputLayer, Layer, Wrapper

from decomon.backward_layers.utils import get_identity_lirpa_shapes
from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
)
from decomon.layers.core import DecomonLayer
from decomon.types import BackendTensor


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
    def call(self, inputs: List[BackendTensor], **kwargs: Any) -> List[BackendTensor]:
        """
        Args:
            inputs

        Returns:

        """
        pass

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Args:
            input_shape

        Returns:

        """
        # generic case: nothing to do before call
        pass

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> List[Tuple[Optional[int], ...]]:
        """Compute expected output shape according to input shape

        Will be called by symbolic calls on Keras Tensors.

        Args:
            input_shape

        Returns:

        """
        if isinstance(self.layer, DecomonLayer):
            decomon_input_shape = [inp.shape for inp in self.layer.input]
            decomon_output_shape = [out.shape for out in self.layer.output]
            keras_output_shape = self.inputs_outputs_spec.get_kerasinputshape_from_inputshapesformode(
                decomon_output_shape
            )
            keras_input_shape = self.inputs_outputs_spec.get_kerasinputshape_from_inputshapesformode(
                decomon_input_shape
            )
        else:  # Keras layer
            keras_output_shape = self.layer.output.shape
            if isinstance(self.layer, InputLayer):
                keras_input_shape = keras_output_shape
            else:
                keras_input_shape = self.layer.input.shape

        batch_size = keras_input_shape[0]
        flattened_keras_output_shape = int(np.prod(keras_output_shape[1:]))  # type: ignore
        flattened_keras_input_shape = int(np.prod(keras_input_shape[1:]))  # type: ignore

        b_shape = batch_size, flattened_keras_output_shape
        w_shape = batch_size, flattened_keras_input_shape, flattened_keras_output_shape

        return [w_shape, b_shape, w_shape, b_shape]

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
