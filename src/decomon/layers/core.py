from abc import ABC, abstractmethod
from enum import Enum
from typing import Union

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


class Option(Enum):
    lagrangian = "lagrangian"
    milp = "milp"


class StaticVariables:
    """Storing static values on the number of input tensors for our layers"""

    def __init__(self, dc_decomp: bool = False, mode: Union[str, ForwardMode] = ForwardMode.HYBRID):
        """
        Args:
            dc_decomp: boolean that indicates whether we return a
                difference of convex decomposition of our layer
            mode: type of Forward propagation (IBP, Forward or Hybrid)
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

    def __init__(
        self,
        convex_domain=None,
        dc_decomp: bool = False,
        mode: str = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs,
    ):
        """
        Args:
            convex_domain: type of convex input domain (None or dict)
            dc_decomp: boolean that indicates whether we return a
            mode: type of Forward propagation (IBP, Forward or Hybrid)
            **kwargs: extra parameters
        difference of convex decomposition of our layer
        """
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

    def get_config(self):
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

    def build(self, input_shape):
        """
        Args:
            input_shape

        Returns:

        """

    @abstractmethod
    def call(self, inputs, **kwargs):
        """
        Args:
            inputs

        Returns:

        """

    def set_linear(self, bool_init):
        self.linear_layer = bool_init

    def get_linear(self):
        return False

    @abstractmethod
    def compute_output_shape(self, input_shape):
        """
        Args:
            input_shape

        Returns:

        """

    def reset_layer(self, layer):
        """
        Args:
            layer

        Returns:

        """

    def join(self, bounds):
        """
        Args:
            bounds

        Returns:

        """
        raise NotImplementedError()

    def freeze_weights(self):
        pass

    def unfreeze_weights(self):
        pass

    def freeze_alpha(self):
        pass

    def unfreeze_alpha(self):
        pass

    def reset_finetuning(self):
        pass

    def shared_weights(self, layer):
        pass

    def split_kwargs(self, **kwargs):
        # necessary for InputLayer
        pass

    def set_back_bounds(self, has_backward_bounds):
        pass
