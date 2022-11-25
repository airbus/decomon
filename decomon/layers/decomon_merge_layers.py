import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Add,
    Average,
    Concatenate,
    Dot,
    Input,
    Lambda,
    Maximum,
    Minimum,
    Multiply,
    Subtract,
)

from .core import F_FORWARD, F_HYBRID, F_IBP, DecomonLayer
from .utils import (
    add,
    broadcast,
    maximum,
    minus,
    multiply,
    permute_dimensions,
    substract,
)

##### Merge Layer ####


class DecomonAdd(Add, DecomonLayer):
    """
    LiRPA implementation of Add layers.
    See Keras official documentation for further details on the Add operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)

    def build(self, input_shape):
        n_comp = self.nb_tensors
        input_shape_y = input_shape[::n_comp]
        super().build(input_shape_y)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]

        if self.mode == F_IBP.name:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == F_HYBRID.name:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == F_FORWARD.name:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [F_IBP.name, F_HYBRID.name]:
            output_u = sum(inputs_u)
            output_l = sum(inputs_l)
        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            output_b_u = sum(inputs_b_u)
            output_b_l = sum(inputs_b_l)

            output_w_u = sum(inputs_w_u)
            output_w_l = sum(inputs_w_l)

        if self.mode == F_IBP.name:
            output = [output_u, output_l]
        elif self.mode == F_FORWARD.name:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == F_HYBRID.name:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output

    def call_dumb(self, inputs):

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors
        if self.mode in [F_HYBRID.name, F_FORWARD.name]:
            input_x = inputs[0]

        inputs_ = [inputs[i * n_comp : (i + 1) * n_comp] for i in range(int(len(inputs) / n_comp))]

        if self.mode == F_HYBRID.name:
            inputs_u = [inputs_[j][1] for j in range(len(inputs_))]
            inputs_l = [inputs_[j][4] for j in range(len(inputs_))]

            inputs_wu = [inputs_[j][2] for j in range(len(inputs_))]
            inputs_bu = [inputs_[j][3] for j in range(len(inputs_))]

            inputs_wl = [inputs_[j][5] for j in range(len(inputs_))]
            inputs_bl = [inputs_[j][6] for j in range(len(inputs_))]
        else:
            raise NotImplementedError()

        return [
            input_x,
            inputs_u[0] + inputs_u[1],
            inputs_wu[0] + inputs_wu[1],
            inputs_bu[0] + inputs_bu[1],
            inputs_l[0] + inputs_l[1],
            inputs_wl[0] + inputs_wl[1],
            inputs_bl[0] + inputs_bl[1],
        ]


class DecomonAverage(Average, DecomonLayer):
    """
    LiRPA implementation of Average layers.
    See Keras official documentation for further details on the Average operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)
        self.op = Lambda(lambda x: sum(x) / len(x))

    def compute_output_shape(self, input_shape):

        return input_shape

    def build(self, input_shape):
        n_comp = self.nb_tensors
        input_shape_y = input_shape[::n_comp]
        super().build(input_shape_y)

    def call(self, inputs):

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            output_x = inputs[0]
        if self.mode == F_IBP.name:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == F_HYBRID.name:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == F_FORWARD.name:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [F_IBP.name, F_HYBRID.name]:
            output_u = self.op(inputs_u)
            output_l = self.op(inputs_l)
        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op(inputs_w_u)
            output_w_l = self.op(inputs_w_l)

        if self.mode == F_IBP.name:
            output = [output_u, output_l]
        elif self.mode == F_FORWARD.name:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == F_HYBRID.name:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return output


class DecomonSubtract(DecomonLayer):
    """
    LiRPA implementation of Subtract layers.
    See Keras official documentation for further details on the Subtract operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)

    def compute_output_shape(self, input_shape):

        return input_shape

    def call(self, inputs, **kwargs):

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
    """
    LiRPA implementation of Minimum layers.
    See Keras official documentation for further details on the Minimum operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)

    def compute_output_shape(self, input_shape):

        return input_shape

    def call(self, inputs, **kwargs):

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
    """
    LiRPA implementation of Maximum layers.
    See Keras official documentation for further details on the Maximum operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):

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
    """
    LiRPA implementation of Concatenate layers.
    See Keras official documentation for further details on the Concatenate operator

    """

    def __init__(self, axis=-1, mode=F_HYBRID.name, **kwargs):
        super().__init__(axis=axis, mode=mode, **kwargs)

        self.op = super().call
        if self.axis == -1:
            self.op_w = self.op
        else:
            self.op_w = Concatenate(axis=self.axis + 1)

    def compute_output_shape(self, input_shape):

        return input_shape

    def build(self, input_shape):
        n_comp = self.nb_tensors
        if self.mode == F_IBP.name:
            input_shape_y = input_shape[::n_comp]
        elif self.mode == F_HYBRID.name:
            input_shape_y = input_shape[1::n_comp]
        elif self.mode == F_FORWARD.name:
            input_shape_y = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_y)

    def call(self, inputs):
        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # splits the inputs
        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            inputs_x = inputs[::n_comp]
            output_x = inputs_x[0]
        if self.mode == F_IBP.name:
            inputs_u = inputs[::n_comp]
            inputs_l = inputs[1::n_comp]
        elif self.mode == F_HYBRID.name:
            inputs_u = inputs[1::n_comp]
            inputs_w_u = inputs[2::n_comp]
            inputs_b_u = inputs[3::n_comp]
            inputs_l = inputs[4::n_comp]
            inputs_w_l = inputs[5::n_comp]
            inputs_b_l = inputs[6::n_comp]
        elif self.mode == F_FORWARD.name:
            inputs_w_u = inputs[1::n_comp]
            inputs_b_u = inputs[2::n_comp]
            inputs_w_l = inputs[3::n_comp]
            inputs_b_l = inputs[4::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [F_IBP.name, F_HYBRID.name]:
            output_u = self.op(inputs_u)
            output_l = self.op(inputs_l)
        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            output_b_u = self.op(inputs_b_u)
            output_b_l = self.op(inputs_b_l)

            output_w_u = self.op_w(inputs_w_u)
            output_w_l = self.op_w(inputs_w_l)

        if self.mode == F_IBP.name:
            output = [output_u, output_l]
        elif self.mode == F_FORWARD.name:
            output = [output_x, output_w_u, output_b_u, output_w_l, output_b_l]
        elif self.mode == F_HYBRID.name:
            output = [output_x, output_u, output_w_u, output_b_u, output_l, output_w_l, output_b_l]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        return output


class DecomonMultiply(Multiply, DecomonLayer):
    """
    LiRPA implementation of Multiply layers.
    See Keras official documentation for further details on the Multiply operator

    """

    def __init__(self, mode=F_HYBRID.name, **kwargs):
        super().__init__(mode=mode, **kwargs)

    def build(self, input_shape):

        n_comp = self.nb_tensors
        if self.mode == F_IBP.name:
            input_shape_ = input_shape[::n_comp]
        elif self.mode == F_HYBRID.name:
            input_shape_ = input_shape[1::n_comp]
        elif self.mode == F_FORWARD.name:
            input_shape_ = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_)

    def call(self, inputs):

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
    """
    LiRPA implementation of Dot layers.
    See Keras official documentation for further details on the Dot operator

    """

    def __init__(self, axes=(-1, -1), mode=F_HYBRID.name, **kwargs):
        super().__init__(axes=axes, mode=mode, **kwargs)
        self.axes = axes

    def build(self, input_shape):

        n_comp = self.nb_tensors
        if self.mode == F_IBP.name:
            input_shape_ = input_shape[::n_comp]
        elif self.mode == F_HYBRID.name:
            input_shape_ = input_shape[1::n_comp]
        elif self.mode == F_FORWARD.name:
            input_shape_ = input_shape[2::n_comp]
        else:
            raise ValueError(f"Unknown mode {self.mode}")
        super().build(input_shape_)

    def call(self, inputs):

        if self.dc_decomp:
            raise NotImplementedError()

        n_comp = self.nb_tensors

        # permute dimensions and reshape
        inputs_0 = inputs[:n_comp]
        inputs_1 = inputs[n_comp:]

        if self.mode == F_IBP.name:
            n_0 = len(inputs_0[0].shape) - 2
            n_1 = len(inputs_1[0].shape) - 2
        elif self.mode in [F_FORWARD.name, F_HYBRID.name]:
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

        if self.mode == F_IBP.name:
            u, l = inputs_[: self.nb_tensors]
        elif self.mode == F_FORWARD.name:
            x, w_u, b_u, w_l, b_l = inputs_[: self.nb_tensors]
        elif self.mode == F_HYBRID.name:
            x, u, w_u, b_u, l, w_l, b_l = inputs_[: self.nb_tensors]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if self.mode in [F_IBP.name, F_HYBRID.name]:
            u_ = K.sum(u, 1)
            l_ = K.sum(l, 1)

        if self.mode in [F_FORWARD.name, F_HYBRID.name]:
            w_u_ = K.sum(w_u, 2)
            b_u_ = K.sum(b_u, 1)
            w_l_ = K.sum(w_l, 2)
            b_l_ = K.sum(b_l, 1)

        if self.mode == F_IBP.name:
            outputs = [u_, l_]
        elif self.mode == F_FORWARD.name:
            outputs = [x, w_u_, b_u_, w_l_, b_l_]
        elif self.mode == F_HYBRID.name:
            outputs = [x, u_, w_u_, b_u_, l_, w_l_, b_l_]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return outputs


def to_monotonic_merge(
    layer,
    input_dim,
    dc_decomp=False,
    convex_domain=None,
    finetune=False,
    IBP=True,
    forward=True,
):
    """Transform a standard Merge layer into a Decomon layer.

    Type of layer is tested to know how to transform it into a MonotonicLayer of the good type.
    If type is not treated yet, raises an TypeError

    :param layer: a Keras Layer
    :param input_dim: either an integer or a tuple that represents the dim of the input convex domain
    :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
    :param convex_domain: the type of convex domain
    :param IBP: boolean that indicates whether we propagate constant bounds
    :param forward: boolean that indicates whether we propagate affine bounds
    :return: the associated DecomonLayer
    :raises: TypeError
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

    mode = F_HYBRID.name
    if IBP and not forward:
        mode = F_IBP.name
    if not IBP and forward:
        mode = F_FORWARD.name

    config_layer["mode"] = mode
    config_layer["finetune"] = finetune

    layer_monotonic = globals()[monotonic_class_name].from_config(config_layer)

    input_shape_list = []
    for input_shape in layer.input_shape:
        input_shape_list.append(list(input_shape[1:]))
    input_shape = input_shape_list

    n_input = len(input_shape_list)
    if isinstance(input_dim, tuple):
        x_shape = Input(input_dim)
        input_dim = input_dim[-1]
    else:
        x_shape = Input((input_dim,))

    w_shape = [Input(tuple([input_dim] + input_shape[i])) for i in range(n_input)]
    y_shape = [Input(tuple(input_shape[i])) for i in range(n_input)]

    if mode == F_HYBRID.name:
        input_ = [
            [x_shape, y_shape[i], w_shape[i], y_shape[i], y_shape[i], w_shape[i], y_shape[i]] for i in range(n_input)
        ]
    elif mode == F_IBP.name:
        input_ = [[y_shape[i], y_shape[i]] for i in range(n_input)]
    elif mode == F_FORWARD.name:
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
