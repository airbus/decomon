from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer


class ForwardMode(Enum):
    """The different forward (from input to output) linear based relaxation perturbation analysis."""

    IBP = "ibp"
    """Propagation of constant bounds from input to output."""

    AFFINE = "affine"
    """Propagation of affine bounds from input to output."""

    HYBRID = "hybrid"
    """Propagation of constant and affines bounds from input to output."""


DEEL_LIP = "deel-lip>"


def get_mode(ibp: bool = True, affine: bool = True) -> ForwardMode:

    if ibp:
        if affine:
            return ForwardMode.HYBRID
        else:
            return ForwardMode.IBP
    else:
        return ForwardMode.AFFINE


class StaticVariables:
    """Storing static values on the number of input tensors for our layers"""

    def __init__(self, dc_decomp: bool = False, mode: Union[str, ForwardMode] = ForwardMode.HYBRID):
        """
        Args:
            dc_decomp: boolean that indicates whether we return a
                difference of convex decomposition of our layer
            mode: type of Forward propagation (ibp, affine, or hybrid)
        gradient
        """

        self.mode = ForwardMode(mode)

        if self.mode == ForwardMode.HYBRID:
            nb_tensors = 7
        elif self.mode == ForwardMode.IBP:
            nb_tensors = 2
        elif self.mode == ForwardMode.AFFINE:
            nb_tensors = 5
        else:
            raise NotImplementedError(f"unknown forward mode {mode}")

        if dc_decomp:
            nb_tensors += 2

        self.nb_tensors = nb_tensors


class DecomonLayer(ABC, Layer):
    """Abstract class that contains the common information of every implemented layers for Forward LiRPA"""

    _trainable_weights: List[tf.Variable]

    @property
    @abstractmethod
    def original_keras_layer_class(self) -> Type[Layer]:
        """The keras layer class from which this class is the decomon equivalent."""
        pass

    def __init__(
        self,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            convex_domain: type of convex input domain (None or dict)
            dc_decomp: boolean that indicates whether we return a
            mode: type of Forward propagation (ibp, affine, or hybrid)
            **kwargs: extra parameters
        difference of convex decomposition of our layer
        """
        kwargs.pop("slope", None)  # remove it if not used by the decomon layer
        super().__init__(**kwargs)

        if convex_domain is None:
            convex_domain = {}
        self.nb_tensors = StaticVariables(dc_decomp, mode).nb_tensors
        self.dc_decomp = dc_decomp
        self.convex_domain = convex_domain
        self.mode = ForwardMode(mode)
        self.finetune = finetune  # extra optimization with hyperparameters
        self.frozen_weights = False
        self.frozen_alpha = False
        self.shared = shared
        self.fast = fast
        self.init_layer = False
        self.linear_layer = False
        self.has_backward_bounds = False  # optimizing Forward LiRPA for adversarial perturbation

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dc_decomp": self.dc_decomp,
                "shared": self.shared,
                "fast": self.fast,
                "finetune": self.finetune,
                "mode": self.mode,
                "convex_domain": self.convex_domain,
            }
        )
        return config

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        """
        Args:
            input_shape

        Returns:

        """
        # generic case: call build from underlying keras layer with the proper intput_shape
        y_input_shape = input_shape[-1]
        self.original_keras_layer_class.build(self, y_input_shape)

    @abstractmethod
    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        """
        Args:
            inputs

        Returns:

        """

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        """
        Args:
            input_shape

        Returns:

        """
        return self.original_keras_layer_class.compute_output_shape(self, input_shape)

    def reset_layer(self, layer: Layer) -> None:
        """
        Args:
            layer

        Returns:

        """

    def join(self, bounds: List[tf.Tensor]) -> List[tf.Tensor]:
        """
        Args:
            bounds

        Returns:

        """
        raise NotImplementedError()

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

    def share_weights(self, layer: Layer) -> None:
        pass

    def split_kwargs(self, **kwargs: Any) -> None:
        # necessary for InputLayer
        pass

    def set_back_bounds(self, has_backward_bounds: bool) -> None:
        pass
