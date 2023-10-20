from typing import Any, Dict, List, Optional, Tuple, Union

import keras_core as keras
import keras_core.ops as K
from keras_core.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Lambda,
    Layer,
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

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int]]]) -> List[Tuple[Optional[int]]]:
        """Compute output shapes from input shapes.

        By default, we assume that all inputs will be merged into "one" (still a list of tensors though).

        """
        input_shapes_list = self.inputs_outputs_spec.split_inputsformode_to_merge(input_shape)
        return input_shapes_list[0]

    def compute_output_spec(self, *args, **kwargs):
        return Layer.compute_output_spec(self, *args, **kwargs)

    def build(self, input_shape: List[Tuple[Optional[int]]]) -> None:
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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        # splits the inputs
        (
            inputs_x,
            inputs_u_c,
            inputs_w_u,
            inputs_b_u,
            inputs_l_c,
            inputs_w_l,
            inputs_b_l,
            inputs_h,
            inputs_g,
        ) = self.inputs_outputs_spec.get_fullinputs_by_type_from_inputsformode_to_merge(inputs)
        x_out = inputs_x[0]
        dtype = x_out.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        # outputs
        if self.ibp:
            u_c_out = sum(inputs_u_c)
            l_c_out = sum(inputs_l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = sum(inputs_b_u)
            b_l_out = sum(inputs_b_l)
            w_u_out = sum(inputs_w_u)
            w_l_out = sum(inputs_w_l)
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x_out, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )


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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        # splits the inputs
        (
            inputs_x,
            inputs_u_c,
            inputs_w_u,
            inputs_b_u,
            inputs_l_c,
            inputs_w_l,
            inputs_b_l,
            inputs_h,
            inputs_g,
        ) = self.inputs_outputs_spec.get_fullinputs_by_type_from_inputsformode_to_merge(inputs)
        x_out = inputs_x[0]
        dtype = x_out.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        # outputs
        if self.ibp:
            u_c_out = self.op(inputs_u_c)
            l_c_out = self.op(inputs_l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = self.op(inputs_b_u)
            b_l_out = self.op(inputs_b_l)
            w_u_out = self.op(inputs_w_u)
            w_l_out = self.op(inputs_w_l)
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x_out, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )


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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)

        # check number of inputs
        if len(inputs_list) != 2:
            raise ValueError("This layer is intended to merge only 2 layers.")

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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)
        # look at minus the input to apply maximum
        inputs_list = [
            minus(single_inputs, mode=self.mode, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain)
            for single_inputs in inputs_list
        ]

        #  check number of inputs
        if len(inputs_list) == 1:  # nothing to merge
            return inputs
        else:
            output = maximum(
                inputs_list[0],
                inputs_list[1],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )
            for j in range(2, len(inputs_list)):
                output = maximum(
                    output,
                    inputs_list[j],
                    dc_decomp=self.dc_decomp,
                    perturbation_domain=self.perturbation_domain,
                    mode=self.mode,
                )

            return minus(output, mode=self.mode, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain)


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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)

        #  check number of inputs
        if len(inputs_list) == 1:  # nothing to merge
            return inputs
        else:
            output = maximum(
                inputs_list[0],
                inputs_list[1],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )
            for j in range(2, len(inputs_list)):
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

        def func(inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
            return Concatenate.call(self, inputs)

        self.op = func
        if self.axis == -1:
            self.op_w = self.op
        else:
            self.op_w = Concatenate(axis=self.axis + 1)

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        # splits the inputs
        (
            inputs_x,
            inputs_u_c,
            inputs_w_u,
            inputs_b_u,
            inputs_l_c,
            inputs_w_l,
            inputs_b_l,
            inputs_h,
            inputs_g,
        ) = self.inputs_outputs_spec.get_fullinputs_by_type_from_inputsformode_to_merge(inputs)
        x_out = inputs_x[0]
        dtype = x_out.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        # outputs
        if self.ibp:
            u_c_out = self.op(inputs_u_c)
            l_c_out = self.op(inputs_l_c)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            b_u_out = self.op(inputs_b_u)
            b_l_out = self.op(inputs_b_l)
            w_u_out = self.op_w(inputs_w_u)
            w_l_out = self.op_w(inputs_w_l)
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x_out, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )


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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)

        #  check number of inputs
        if len(inputs_list) == 1:  # nothing to merge
            return inputs
        else:
            output = multiply(
                inputs_list[0],
                inputs_list[1],
                dc_decomp=self.dc_decomp,
                perturbation_domain=self.perturbation_domain,
                mode=self.mode,
            )
            for j in range(2, len(inputs_list)):
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

    def call(self, inputs: List[keras.KerasTensor], **kwargs: Any) -> List[keras.KerasTensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        inputs_list = self.inputs_outputs_spec.split_inputsformode_to_merge(inputs)

        #  check number of inputs
        if len(inputs_list) == 1:  # nothing to merge
            return inputs
        elif len(inputs_list) == 2:
            inputs_0, inputs_1 = inputs_list
        else:
            raise NotImplementedError("This layer is not implemented to merge more than 2 layers.")

        input_shape_0 = self.inputs_outputs_spec.get_input_shape(inputs_0)
        input_shape_1 = self.inputs_outputs_spec.get_input_shape(inputs_1)
        n_0 = len(input_shape_0) - 2
        n_1 = len(input_shape_1) - 2

        inputs_0 = permute_dimensions(inputs_0, self.axes[0], mode=self.mode, dc_decomp=self.dc_decomp)
        inputs_1 = permute_dimensions(inputs_1, self.axes[1], mode=self.mode, dc_decomp=self.dc_decomp)
        inputs_0 = broadcast(
            inputs_0, n_1, -1, mode=self.mode, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain
        )
        inputs_1 = broadcast(
            inputs_1, n_0, 2, mode=self.mode, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain
        )
        outputs_multiply = multiply(
            inputs_0, inputs_1, dc_decomp=self.dc_decomp, perturbation_domain=self.perturbation_domain, mode=self.mode
        )

        x, u_c, w_u, b_u, l_c, w_l, b_l, h, g = self.inputs_outputs_spec.get_fullinputs_from_inputsformode(
            outputs_multiply, compute_ibp_from_affine=False
        )
        dtype = x.dtype
        empty_tensor = self.inputs_outputs_spec.get_empty_tensor(dtype=dtype)

        if self.ibp:
            u_c_out = K.sum(u_c, 1)
            l_c_out = K.sum(l_c, 1)
        else:
            u_c_out, l_c_out = empty_tensor, empty_tensor

        if self.affine:
            w_u_out = K.sum(w_u, 2)
            b_u_out = K.sum(b_u, 1)
            w_l_out = K.sum(w_l, 2)
            b_l_out = K.sum(b_l, 1)
        else:
            w_u_out, b_u_out, w_l_out, b_l_out = empty_tensor, empty_tensor, empty_tensor, empty_tensor

        if self.dc_decomp:
            raise NotImplementedError()
        else:
            h_out, g_out = empty_tensor, empty_tensor

        return self.inputs_outputs_spec.extract_outputsformode_from_fulloutputs(
            [x, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out, h_out, g_out]
        )
