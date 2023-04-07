from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Input,
    Lambda,
    Layer,
    Multiply,
)

from decomon.layers.core import DecomonLayer, ForwardMode
from decomon.layers.utils import broadcast, multiply, permute_dimensions
from decomon.utils import ConvexDomainType, maximum, minus, substract

##### Merge Layer ####


class DecomonAdd(Add, DecomonLayer):
    """LiRPA implementation of Add layers.
    See Keras official documentation for further details on the Add operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        n_comp = self.nb_tensors
        input_shape_y = input_shape[::n_comp]
        super().build(input_shape_y)

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]

        if self.mode == ForwardMode.IBP:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
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
            output_u = sum(inputs_u)
            output_l = sum(inputs_l)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = sum(inputs_b_u)
            output_b_l = sum(inputs_b_l)

            output_w_u = sum(inputs_w_u)
            output_w_l = sum(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u, output_l]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output


class DecomonAverage(Average, DecomonLayer):
    """LiRPA implementation of Average layers.
    See Keras official documentation for further details on the Average operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )
        self.op = Lambda(lambda x: sum(x) / len(x))

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:

        return input_shape

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        n_comp = self.nb_tensors
        input_shape_y = input_shape[::n_comp]
        super().build(input_shape_y)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_x = inputs[0]
        if self.mode == ForwardMode.IBP:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
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
            output_u = self.op(inputs_u)
            output_l = self.op(inputs_l)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op(inputs_w_u)
            output_w_l = self.op(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u, output_l]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output


class DecomonSubtract(DecomonLayer):
    """LiRPA implementation of Subtract layers.
    See Keras official documentation for further details on the Subtract operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:

        return input_shape

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        # splits the inputs
        n_comp = self.nb_tensors
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(inputs) // n_comp)]

        output = substract(
            inputs_list[0], inputs_list[1], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
        )
        return output


class DecomonMinimum(DecomonLayer):
    """LiRPA implementation of Minimum layers.
    See Keras official documentation for further details on the Minimum operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:

        return input_shape

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
            inputs_list[0], inputs_list[1], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
        )
        for j in range(2, len(inputs) // n_comp):
            output = maximum(
                output, inputs_list[j], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
            )

        return minus(output, mode=self.mode)


class DecomonMaximum(DecomonLayer):
    """LiRPA implementation of Maximum layers.
    See Keras official documentation for further details on the Maximum operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:
        return input_shape

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
            inputs_list[0], inputs_list[1], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
        )
        for j in range(2, len(inputs) // n_comp):
            output = maximum(
                output, inputs_list[j], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
            )

        return output


class DecomonConcatenate(Concatenate, DecomonLayer):
    """LiRPA implementation of Concatenate layers.
    See Keras official documentation for further details on the Concatenate operator

    """

    def __init__(
        self,
        axis: int = -1,
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            axis=axis,
            convex_domain=convex_domain,
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

    def compute_output_shape(self, input_shape: List[tf.TensorShape]) -> List[tf.TensorShape]:

        return input_shape

    def build(self, input_shape: List[tf.TensorShape]) -> None:
        n_comp = self.nb_tensors
        if self.mode == ForwardMode.IBP:
            input_shape_y = input_shape[::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            input_shape_y = input_shape[1::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            input_shape_y = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_y)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:
        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]
        if self.mode == ForwardMode.IBP:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
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
            output_u = self.op(inputs_u)
            output_l = self.op(inputs_l)
        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op_w(inputs_w_u)
            output_w_l = self.op_w(inputs_w_l)

        if self.mode == ForwardMode.IBP:
            output = [output_u, output_l]
        elif self.mode == ForwardMode.AFFINE:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == ForwardMode.HYBRID:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return output


class DecomonMultiply(Multiply, DecomonLayer):
    """LiRPA implementation of Multiply layers.
    See Keras official documentation for further details on the Multiply operator

    """

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
        super().__init__(
            convex_domain=convex_domain,
            dc_decomp=dc_decomp,
            mode=mode,
            finetune=finetune,
            shared=shared,
            fast=fast,
            **kwargs,
        )

    def build(self, input_shape: List[tf.TensorShape]) -> None:

        n_comp = self.nb_tensors
        if self.mode == ForwardMode.IBP:
            input_shape_ = input_shape[::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            input_shape_ = input_shape[1::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            input_shape_ = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_)

    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(inputs) // n_comp)]

        output = multiply(
            inputs_list[0], inputs_list[1], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
        )
        for j in range(2, len(inputs) // n_comp):
            output = multiply(
                output, inputs_list[j], dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
            )

        return output


class DecomonDot(Dot, DecomonLayer):
    """LiRPA implementation of Dot layers.
    See Keras official documentation for further details on the Dot operator

    """

    def __init__(
        self,
        axes: Union[int, Tuple[int, int]] = (-1, -1),
        convex_domain: Optional[Dict[str, Any]] = None,
        dc_decomp: bool = False,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        finetune: bool = False,
        shared: bool = False,
        fast: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            axes=axes,
            convex_domain=convex_domain,
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

    def build(self, input_shape: List[tf.TensorShape]) -> None:

        n_comp = self.nb_tensors
        if self.mode == ForwardMode.IBP:
            input_shape_ = input_shape[::n_comp]
        elif self.mode == ForwardMode.HYBRID:
            input_shape_ = input_shape[1::n_comp]
        elif self.mode == ForwardMode.AFFINE:
            input_shape_ = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_)

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

        input_0_0 = permute_dimensions(inputs_0, self.axes[0], mode=self.mode)
        input_1_0 = permute_dimensions(inputs_1, self.axes[1], mode=self.mode)

        inputs_0_ = broadcast(input_0_0, n_1, -1, mode=self.mode)
        inputs_1_ = broadcast(input_1_0, n_0, 2, mode=self.mode)

        inputs_ = multiply(
            inputs_0_, inputs_1_, dc_decomp=self.dc_decomp, convex_domain=self.convex_domain, mode=self.mode
        )

        if self.mode == ForwardMode.IBP:
            u, l = inputs_[: self.nb_tensors]
        elif self.mode == ForwardMode.AFFINE:
            x, w_u, b_u, w_l, b_l = inputs_[: self.nb_tensors]
        elif self.mode == ForwardMode.HYBRID:
            x, u, w_u, b_u, l, w_l, b_l = inputs_[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [ForwardMode.IBP, ForwardMode.HYBRID]:
            u_ = K.sum(u, 1)
            l_ = K.sum(l, 1)

        if self.mode in [ForwardMode.AFFINE, ForwardMode.HYBRID]:
            w_u_ = K.sum(w_u, 2)
            b_u_ = K.sum(b_u, 1)
            w_l_ = K.sum(w_l, 2)
            b_l_ = K.sum(b_l, 1)

        if self.mode == ForwardMode.IBP:
            outputs = [u_, l_]
        elif self.mode == ForwardMode.AFFINE:
            outputs = [x, w_u_, b_u_, w_l_, b_l_]
        elif self.mode == ForwardMode.HYBRID:
            outputs = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return outputs


def to_decomon_merge(
    layer: Layer,
    input_dim: int,
    dc_decomp: bool = False,
    convex_domain: Optional[Dict[str, Any]] = None,
    finetune: bool = False,
    ibp: bool = True,
    affine: bool = True,
) -> List[DecomonLayer]:
    """Transform a standard Merge layer into a Decomon layer.

    Type of layer is tested to know how to transform it into a MonotonicLayer of the good type.
    If type is not treated yet, raises an TypeError

    Args:
        layer: a Keras Layer
        input_dim: an integer that represents the dim
            of the input convex domain
        dc_decomp: boolean that indicates whether we return a difference
            of convex decomposition of our layer
        convex_domain: the type of convex domain
        ibp: boolean that indicates whether we propagate constant bounds
        affine: boolean that indicates whether we propagate affine
            bounds

    Returns:
        the associated DecomonLayer
    """

    # get class name
    if convex_domain is None:
        convex_domain = {}
    class_name = layer.__class__.__name__
    # check if layer has a built argument that built is set to True
    if hasattr(layer, "built"):
        if not layer.built:
            raise ValueError(f"the layer {layer.name} has not been built yet")

    monotonic_class_name = f"Decomon{class_name}"
    config_layer = layer.get_config()
    config_layer["name"] = layer.name + "_monotonic"
    config_layer["dc_decomp"] = dc_decomp
    config_layer["convex_domain"] = convex_domain

    mode = ForwardMode.HYBRID
    if ibp and not affine:
        mode = ForwardMode.IBP
    if not ibp and affine:
        mode = ForwardMode.AFFINE

    config_layer["mode"] = mode
    config_layer["finetune"] = finetune

    layer_monotonic = globals()[monotonic_class_name].from_config(config_layer)

    input_shape_list = []
    for input_shape in layer.input_shape:
        input_shape_list.append(list(input_shape[1:]))
    input_shape = input_shape_list

    n_input = len(input_shape_list)
    if len(convex_domain) == 0 or convex_domain["name"] == ConvexDomainType.BOX:
        x_shape = Input((2, input_dim), dtype=layer.dtype)
    else:
        x_shape = Input((input_dim,), dtype=layer.dtype)

    w_shape = [Input(tuple([input_dim] + input_shape[i])) for i in range(n_input)]
    y_shape = [Input(tuple(input_shape[i])) for i in range(n_input)]

    if mode == ForwardMode.HYBRID:
        input_ = [
            [x_shape, y_shape[i], w_shape[i], y_shape[i], y_shape[i], w_shape[i], y_shape[i]] for i in range(n_input)
        ]
    elif mode == ForwardMode.IBP:
        input_ = [[y_shape[i], y_shape[i]] for i in range(n_input)]
    elif mode == ForwardMode.AFFINE:
        input_ = [[x_shape, w_shape[i], y_shape[i], w_shape[i], y_shape[i]] for i in range(n_input)]
    else:
        raise ValueError(f"Unknown mode {mode}")
    if dc_decomp:
        raise NotImplementedError()

    input_list = []
    for i in range(n_input):
        input_list += input_[i]

    layer_monotonic(input_list)
    layer_monotonic.reset_layer(layer)

    return [layer_monotonic]
