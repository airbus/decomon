from __future__ import absolute_import

from abc import ABC, abstractmethod

from tensorflow.keras.layers import Layer

from ..types import Optional

#  the different forward (from input to output) linear based relaxation perturbation analysis

# propgation of constant bounds from input to output
class F_IBP:
    name = "ibp"


# propagation of affine bounds from input to output
class F_FORWARD:
    name = "forward"


# propagation of constant and affines bounds from input to output
class F_HYBRID:
    name = "hybrid"


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Grid:
    name = "grid"


class Vertex:
    name = "vertex"  # convex set represented by its vertices
    # (no verification is proceeded to assess that the set is convex)


class DEEL_LIP:
    name = "deel-lip>"


class Option:
    lagrangian = "lagrangian"
    milp = "milp"


class StaticVariables:
    """
    Storing static values on the number of input tensors for our layers
    """

    def __init__(self, dc_decomp: Optional[bool] = False, mode: Optional[str] = F_HYBRID.name):
        """

        :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
        gradient
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        """

        self.mode = mode

        if self.mode == F_HYBRID.name:
            nb_tensors = 7
        elif self.mode == F_IBP.name:
            nb_tensors = 2
        elif self.mode == F_FORWARD.name:
            nb_tensors = 5
        else:
            raise NotImplementedError(f"unknown forward mode {mode}")

        if dc_decomp:
            nb_tensors += 2

        self.nb_tensors = nb_tensors


class DecomonLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers for Forward LiRPA
    """

    def __init__(
        self,
        convex_domain=None,
        dc_decomp: Optional[bool] = False,
        mode: Optional[str] = F_HYBRID.name,
        finetune: Optional[bool] = False,
        shared: Optional[bool] = False,
        fast: Optional[bool] = True,
        **kwargs,
    ):
        """

        :param convex_domain: type of convex input domain (None or dict)
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param mode: type of Forward propagation (IBP, Forward or Hybrid)
        :param kwargs: extra parameters
        """
        super().__init__(**kwargs)

        if convex_domain is None:
            convex_domain = {}
        self.nb_tensors = StaticVariables(dc_decomp, mode).nb_tensors
        self.dc_decomp = dc_decomp
        self.convex_domain = convex_domain
        self.mode = mode
        self.finetune = finetune  # extra optimization with hyperparameters
        self.frozen_weights = False
        self.frozen_alpha = False
        self.shared = shared
        self.fast = fast
        self.init_layer = False
        self.linear_layer = False
        self.has_backward_bounds = False  # optimizing Forward LiRPA for adversarial perturbation

    def build(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        pass

    @abstractmethod
    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :return:
        """
        pass

    def set_linear(self, bool_init):
        self.linear_layer = bool_init

    def get_linear(self):
        return False

    @abstractmethod
    def compute_output_shape(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        pass

    def reset_layer(self, layer):
        """

        :param layer:
        :return:
        """
        pass

    def join(self, bounds):
        """

        :param bounds:
        :return:
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
        # 'mode', 'dc_decomp', 'convex_domain', 'finetune', 'shared', 'fast'])
        pass

    def set_back_bounds(self, has_backward_bounds):
        pass
