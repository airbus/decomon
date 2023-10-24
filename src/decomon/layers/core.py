from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import keras
from keras.layers import Layer

from decomon.core import (
    BoxDomain,
    ForwardMode,
    InputsOutputsSpec,
    PerturbationDomain,
    get_affine,
    get_ibp,
)
from decomon.keras_utils import reset_layer


class DecomonLayer(ABC, Layer):
    """Abstract class that contains the common information of every implemented layers for Forward LiRPA"""

    _trainable_weights: List[keras.Variable]

    @property
    @abstractmethod
    def original_keras_layer_class(self) -> Type[Layer]:
        """The keras layer class from which this class is the decomon equivalent."""
        pass

    def __init__(
        self,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            perturbation_domain: type of convex input domain (None or dict)
            dc_decomp: boolean that indicates whether we return a
            mode: type of Forward propagation (ibp, affine, or hybrid)
            **kwargs: extra parameters
        difference of convex decomposition of our layer
        """
        kwargs.pop("slope", None)  # remove it if not used by the decomon layer
        super().__init__(**kwargs)

        if perturbation_domain is None:
            perturbation_domain = BoxDomain()
        self.inputs_outputs_spec = InputsOutputsSpec(
            dc_decomp=dc_decomp, mode=mode, perturbation_domain=perturbation_domain
        )
        self.nb_tensors = self.inputs_outputs_spec.nb_tensors
        self.dc_decomp = dc_decomp
        self.perturbation_domain = perturbation_domain
        self.mode = ForwardMode(mode)
        self.finetune = finetune  # extra optimization with hyperparameters
        self.frozen_weights = False
        self.frozen_alpha = False
        self.shared = shared
        self.fast = fast
        self.has_backward_bounds = False  # optimizing Forward LiRPA for adversarial perturbation

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
                "dc_decomp": self.dc_decomp,
                "shared": self.shared,
                "fast": self.fast,
                "finetune": self.finetune,
                "mode": self.mode,
                "perturbation_domain": self.perturbation_domain,
            }
        )
        return config

    def build(self, input_shape: List[Tuple[Optional[int]]]) -> None:
        """
        Args:
            input_shape

        Returns:

        """
        # generic case: call build from underlying keras layer with the proper intput_shape
        y_input_shape = input_shape[-1]
        self.original_keras_layer_class.build(self, y_input_shape)

    @abstractmethod
    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        """
        Args:
            inputs

        Returns:

        """

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int]]]) -> List[Tuple[Optional[int]]]:
        """Compute expected output shape according to input shape

        Will be called by symbolic calls on Keras Tensors.

        - We use the original (Keras) layer compute_output_shape() if available to update accordingly the input shapes.
        - Else we simply return the input shapes

        Args:
            input_shape

        Returns:

        """
        y_shape = self.inputs_outputs_spec.get_kerasinputshape_from_inputshapesformode(
            input_shape
        )  # input shape for the original layer
        try:
            y_out_shape = self.original_keras_layer_class.compute_output_shape(
                self, y_shape
            )  # output shape of the original layer
        except NotImplementedError:
            # no compute_output_shape existing (e.g. InputLayer)
            return input_shape
        else:
            (
                x_shape,
                u_c_shape,
                w_u_shape,
                b_u_shape,
                l_c_shape,
                w_l_shape,
                b_l_shape,
                h_shape,
                g_shape,
            ) = self.inputs_outputs_spec.get_fullinputshapes_from_inputshapesformode(input_shape)
            # we now the original output shape => deduce the decomon output shapes
            y_out_shape_wo_batchsize = y_out_shape[1:]
            if self.inputs_outputs_spec.affine:
                model_inputdim = x_shape[-1]
                batchsize = x_shape[0]
                w_out_shape = (batchsize, model_inputdim) + y_out_shape_wo_batchsize
            else:
                w_out_shape = tuple()
            fulloutputshapes = [
                x_shape,
                y_out_shape,
                w_out_shape,
                y_out_shape,
                y_out_shape,
                w_out_shape,
                y_out_shape,
                y_out_shape,
                y_out_shape,
            ]
            return self.inputs_outputs_spec.extract_inputshapesformode_from_fullinputshapes(fulloutputshapes)

    def compute_output_spec(self, *args, **kwargs):
        """Compute output spec from output shape in case of symbolic call."""
        return Layer.compute_output_spec(self, *args, **kwargs)

    def reset_layer(self, layer: Layer) -> None:
        """Reset the weights by using the weights of another (a priori non-decomon) layer.

        It set the weights whose names are listed by `keras_weights_names`.

        Args:
            layer

        Returns:

        """
        weight_names = self.keras_weights_names
        if len(weight_names) > 0:
            reset_layer(new_layer=self, original_layer=layer, weight_names=weight_names)

    @property
    def keras_weights_names(self) -> List[str]:
        """Weights names of the corresponding Keras layer.

        Will be used to decide which weight to take from the keras layer in `reset_layer()`

        """
        return []

    def join(self, bounds: List[keras.KerasTensor]) -> List[keras.KerasTensor]:
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
