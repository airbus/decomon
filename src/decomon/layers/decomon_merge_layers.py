from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Lambda,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
)

from decomon.core import ForwardMode, PerturbationDomain
from decomon.layers.core import DecomonLayer
from decomon.layers.utils import broadcast, multiply, permute_dimensions
from decomon.utils import maximum, minus, subtract

##### Merge Layer ####


class DecomonMerge(DecomonLayer):
    """Base class for Decomon layers based on Mergind Keras layers."""

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        n_comp = self.nb_tensors
        input_shape_y = input_shape[n_comp - 1 :: n_comp]
        self.original_keras_layer_class.build(self, input_shape_y)


class DecomonAdd(DecomonMerge, Add):
    """LiRPA implementation of Add layers.
    See Keras official documentation for further details on the Add operator

    """

    original_keras_layer_class = Add

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]

        if self.mode == ForwardMode.IBP:
            inputs_u_c = inputs[::n_comp]
            inputs_l_c = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u_c = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l_c = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            output_u_c = sum(inputs_u_c)
            output_l_c = sum(inputs_l_c)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = sum(inputs_b_u)
            output_b_l = sum(inputs_b_l)

            output_w_u = sum(inputs_w_u)
            output_w_l = sum(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u_c, output_l_c]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u_c, output_w_u, output_b_u, output_l_c, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output


class DecomonAverage(DecomonMerge, Average):
    """LiRPA implementation of Average layers.
    See Keras official documentation for further details on the Average operator

    """

    original_keras_layer_class = Average

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.op = Lambda(lambda x: sum(x) / len(x))

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_x = inputs[0]
        if self.mode == ForwardMode.IBP:
            inputs_u_c = inputs[::n_comp]
            inputs_l_c = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u_c = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l_c = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            output_u_c = self.op(inputs_u_c)
            output_l_c = self.op(inputs_l_c)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op(inputs_w_u)
            output_w_l = self.op(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u_c, output_l_c]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u_c, output_w_u, output_b_u, output_l_c, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output


class DecomonSubtract(DecomonMerge, Subtract):
    """LiRPA implementation of Subtract layers.
    See Keras official documentation for further details on the Subtract operator

    """

    original_keras_layer_class = Subtract

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        n_comp = self.nb_tensors
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(inputs) // n_comp)]

        output = subtract(
            inputs_list[0],
            inputs_list[1],
            dc_decomp=self.dc_decomp,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
        )
        return output


class DecomonMinimum(DecomonMerge, Minimum):
    """LiRPA implementation of Minimum layers.
    See Keras official documentation for further details on the Minimum operator

    """

    original_keras_layer_class = Minimum

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # check there is more than one input
        if len(inputs) == n_comp:
            return inputs

        # splits the inputs
        inputs_list = [
            minus(inputs[n_comp * i : n_comp * (i + 1)], mode=self.mode) for i in range(len(inputs) // n_comp)
        ]

        output = maximum(
            inputs_list[0],
            inputs_list[1],
            dc_decomp=self.dc_decomp,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
        )
        for j in range(2, len(inputs) // n_comp):
            output = maximum(
                output,
                inputs_list[j],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )

        return minus(output, mode=self.mode)


class DecomonMaximum(DecomonMerge, Maximum):
    """LiRPA implementation of Maximum layers.
    See Keras official documentation for further details on the Maximum operator

    """

    original_keras_layer_class = Maximum

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # check there is more than one input
        if len(inputs) == n_comp:
            return inputs
        # splits the inputs
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(inputs) // n_comp)]

        output = maximum(
            inputs_list[0],
            inputs_list[1],
            dc_decomp=self.dc_decomp,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
        )
        for j in range(2, len(inputs) // n_comp):
            output = maximum(
                output,
                inputs_list[j],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )

        return output


class DecomonConcatenate(DecomonMerge, Concatenate):
    """LiRPA implementation of Concatenate layers.
    See Keras official documentation for further details on the Concatenate operator

    """

    original_keras_layer_class = Concatenate

    def __init__(
        self,
        axis: int = -1,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            axis=axis,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

        def func(inputs: List[tf.Tensor]) -> tf.Tensor:
            return Concatenate.call(self, inputs)

        self.op = func
        if self.axis == -1:
            self.op_w = self.op
        else:
            self.op_w = Concatenate(axis=self.axis + 1)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]
        if self.mode == ForwardMode.IBP:
            inputs_u_c = inputs[::n_comp]
            inputs_l_c = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u_c = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l_c = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            output_u_c = self.op(inputs_u_c)
            output_l_c = self.op(inputs_l_c)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op_w(inputs_w_u)
            output_w_l = self.op_w(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u_c, output_l_c]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u_c, output_w_u, output_b_u, output_l_c, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return output


class DecomonMultiply(DecomonMerge, Multiply):
    """LiRPA implementation of Multiply layers.
    See Keras official documentation for further details on the Multiply operator

    """

    original_keras_layer_class = Multiply

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
        super().__init__(
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(inputs) // n_comp)]

        output = multiply(
            inputs_list[0],
            inputs_list[1],
            dc_decomp=self.dc_decomp,
            perturbation_domain=self.perturbation_domain,
            mode=self.mode,
        )
        for j in range(2, len(inputs) // n_comp):
            output = multiply(
                output,
                inputs_list[j],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )

        return output


class DecomonDot(DecomonMerge, Dot):
    """LiRPA implementation of Dot layers.
    See Keras official documentation for further details on the Dot operator

    """

    original_keras_layer_class = Dot

    def __init__(
        self,
        axes: Union[int, Tuple[int, int]] = (-1, -1),
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            axes=axes,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        if isinstance(axes, int):
            self.axes = (axes, axes)
        else:
            self.axes = axes

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # permute dimensions and reshape
        inputs_0 = inputs[:n_comp]
        inputs_1 = inputs[n_comp:]

        if self.mode == ForwardMode.IBP:
            n_0 = len(inputs_0[0].shape) - 2
            n_1 = len(inputs_1[0].shape) - 2
        elif self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            n_0 = len(inputs_0[-1].shape) - 2
            n_1 = len(inputs_1[-1].shape) - 2
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        inputs_0 = permute_dimensions(inputs_0, self.axes[0], mode=self.mode, dc_decomp=self.dc_decomp)
        inputs_1 = permute_dimensions(inputs_1, self.axes[1], mode=self.mode, dc_decomp=self.dc_decomp)

        inputs_0 = broadcast(inputs_0, n_1, -1, mode=self.mode, dc_decomp=self.dc_decomp)
        inputs_1 = broadcast(inputs_1, n_0, 2, mode=self.mode, dc_decomp=self.dc_decomp)

        outputs_multiply = multiply(
            inputs_0, inputs_1, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain, mode=self.mode
        )

        if self.mode == ForwardMode.IBP:
            u_c, l_c = outputs_multiply[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x, w_u, b_u, w_l, b_l = outputs_multiply[: self.nb_tensors]
        elif self.mode == ForwardMode.HYBRID:
            x, u_c, w_u, b_u, l_c, w_l, b_l = outputs_multiply[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_c_out = K.sum(u_c, 1)
            l_c_out = K.sum(l_c, 1)

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            w_u_out = K.sum(w_u, 2)
            b_u_out = K.sum(b_u, 1)
            w_l_out = K.sum(w_l, 2)
            b_l_out = K.sum(b_l, 1)

        if self.mode == ForwardMode.IBP:
            outputs = [u_c_out, l_c_out]
        elif self.mode == ForwardMode.AFFINE:
            outputs = [x, w_u_out, b_u_out, w_l_out, b_l_out]
        elif self.mode == ForwardMode.HYBRID:
            outputs = [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return outputs
