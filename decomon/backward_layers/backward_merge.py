from __future__ import absolute_import
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Flatten, Dot, Permute
from decomon.layers.decomon_merge_layers import (
    DecomonSubtract,
    DecomonMaximum,
    DecomonMinimum,
    DecomonAdd,
    DecomonAverage,
    DecomonConcatenate,
    DecomonMultiply,
    DecomonDot,
)
from ..backward_layers.activations import get
from .utils import V_slope, backward_add, backward_substract, backward_maximum, backward_minimum, backward_multiply
from ..layers.core import F_HYBRID, F_FORWARD, F_IBP
from ..layers.utils import split, multiply, permute_dimensions, broadcast


class BackwardAdd(Layer):
    """
    Backward  LiRPA of Add
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardAdd, self).__init__(**kwargs)
        # if not isinstance(layer, DecomonAdd):
        #    raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        # inputs_list = [inputs[n_comp * i:n_comp * (i + 1)] for i in range(n_elem)]

        if n_elem == 1:
            return [[w_out_u, b_out_u, w_out_l, b_out_l]]
        else:
            bounds = []
            input_bounds = []

            for j in np.arange(1, n_elem)[::-1]:
                inputs_1 = x_[n_comp * j : n_comp * (j + 1)]
                if j == 1:
                    inputs_0 = x_[:n_comp]
                else:
                    inputs_0 = self.layer(x_[: n_comp * j])
                if len(bounds) == 0:
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u,
                        b_out_u,
                        w_out_l,
                        b_out_l,
                        convex_domain=self.layer.convex_domain,
                        mode=self.mode,
                    )
                else:
                    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = bounds[-1]
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u_,
                        b_out_u_,
                        w_out_l_,
                        b_out_l_,
                        convex_domain=self.layer.convex_domain,
                        mode=self.mode,
                    )

                input_bounds.append(bounds_1)
                bounds.append(bounds_0)
                if j == 1:
                    input_bounds.append(bounds_0)

        input_bounds = input_bounds[::-1]

        return input_bounds


class BackwardAverage(Layer):
    """
    Backward  LiRPA of Average
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardAverage, self).__init__(**kwargs)
        if not isinstance(layer, DecomonAverage):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode
        self.op = DecomonAdd(mode=self.mode, convex_domain=self.layer.convex_domain, dc_decomp=layer.dc_decomp).call

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        # inputs_list = [inputs[n_comp * i:n_comp * (i + 1)] for i in range(n_elem)]

        if n_elem == 1:
            return [[w_out_u, b_out_u, w_out_l, b_out_l]]
        else:
            bounds = []
            input_bounds = []

            for j in np.arange(1, n_elem)[::-1]:
                inputs_1 = x_[n_comp * j : n_comp * (j + 1)]
                if j == 1:
                    inputs_0 = x_[:n_comp]
                else:
                    inputs_0 = self.op(x_[: n_comp * j])
                if len(bounds) == 0:
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u,
                        b_out_u,
                        w_out_l,
                        b_out_l,
                        convex_domain=self.layer.convex_domain,
                        mode=self.mode,
                    )
                else:
                    w_out_u_, b_out_u_, w_out_l_, b_out_l_ = bounds[-1]
                    bounds_0, bounds_1 = backward_add(
                        inputs_0,
                        inputs_1,
                        w_out_u_,
                        b_out_u_,
                        w_out_l_,
                        b_out_l_,
                        convex_domain=self.layer.convex_domain,
                        mode=self.mode,
                    )

                input_bounds.append(bounds_1)
                bounds.append(bounds_0)
                if j == 1:
                    input_bounds.append(bounds_0)

        input_bounds = input_bounds[::-1]
        input_bounds = [[1.0 / n_elem * elem_i for elem_i in elem] for elem in input_bounds]

        return input_bounds


class BackwardSubtract(Layer):
    """
    Backward  LiRPA of Subtract
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardSubtract, self).__init__(**kwargs)
        if not isinstance(layer, DecomonSubtract):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_substract(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardMaximum(Layer):
    """
    Backward  LiRPA of Maximum
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardMaximum, self).__init__(**kwargs)
        if not isinstance(layer, DecomonMaximum):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_maximum(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardMinimum(Layer):
    """
    Backward  LiRPA of Minimum
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardMinimum, self).__init__(**kwargs)
        if not isinstance(layer, DecomonMinimum):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_minimum(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardConcatenate(Layer):
    """
    Backward  LiRPA of Concatenate
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardConcatenate, self).__init__(**kwargs)
        if not isinstance(layer, DecomonConcatenate):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode
        self.axis = self.layer.axis

    def call(self, inputs):
        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        n_list = [inputs[n_comp * i : n_comp * (i + 1)][0].shape[self.axis] for i in range(len(x_) // n_comp)]
        axis_w = self.axis
        if axis_w != -1:
            axis_w += 1
        w_out_u_list = tf.split(w_out_u, n_list, axis_w)
        w_out_l_list = tf.split(w_out_l, n_list, axis_w)
        b_out_u_list = tf.split(b_out_u, n_list, self.axis)
        b_out_l_list = tf.split(b_out_l, n_list, self.axis)

        bounds = [[w_out_u_list[i], b_out_u_list[i], w_out_l_list[i], b_out_l_list[i]] for i in range(n_elem)]

        return bounds


class BackwardMultiply(Layer):
    """
    Backward  LiRPA of Multiply
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardMultiply, self).__init__(**kwargs)
        if not isinstance(layer, DecomonMultiply):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        return backward_multiply(
            inputs_list[0],
            inputs_list[1],
            w_out_u,
            b_out_u,
            w_out_l,
            b_out_l,
            convex_domain=self.layer.convex_domain,
            mode=self.mode,
        )


class BackwardDot(Layer):
    """
    Backward  LiRPA of Dot
    """

    def __init__(self, layer, slope=V_slope.name, **kwargs):
        super(BackwardDot, self).__init__(**kwargs)
        if not isinstance(layer, DecomonDot):
            raise KeyError()

        self.layer = layer
        self.slope = slope
        self.mode = self.layer.mode
        self.axes = [i for i in self.layer.axes]
        self.op = BackwardAdd(self.layer)
        self.convex_domain = self.layer.convex_domain

        raise NotImplementedError()

    def call(self, inputs):

        x_ = inputs[:-4]
        w_out_u, b_out_u, w_out_l, b_out_l = inputs[-4:]

        n_comp = 4
        if self.mode == F_FORWARD.name:
            n_comp = 6
        if self.mode == F_HYBRID.name:
            n_comp = 8

        n_elem = len(x_) // n_comp
        inputs_list = [inputs[n_comp * i : n_comp * (i + 1)] for i in range(len(x_) // n_comp)]
        if n_elem != 2:
            raise ValueError()

        # permute dimensions and reshape
        inputs_0 = inputs_list[0]
        inputs_1 = inputs_list[1]

        n_0 = len(inputs_0[0].shape) - 2
        n_1 = len(inputs_1[0].shape) - 2

        input_0_0 = permute_dimensions(inputs_0, self.axes[0], mode=self.mode)
        input_1_0 = permute_dimensions(inputs_1, self.axes[1], mode=self.mode)

        inputs_0_ = broadcast(input_0_0, n_1, -1, mode=self.mode)
        inputs_1_ = broadcast(input_1_0, n_0, 2, mode=self.mode)

        import pdb

        pdb.set_trace()

        inputs_ = multiply(
            inputs_0_,
            inputs_1_,
            dc_decomp=self.layer.dc_decomp,
            convex_domain=self.convex_domain,
            mode=self.mode,
        )

        inputs_add_ = split(inputs_, axis=self.axes[0], mode=self.mode)
        inputs_add = []
        for elem in inputs_add_:
            inputs_add += elem

        bounds = self.op.call(inputs_add + inputs[-4:])
        n = len(inputs_add)
        bounds_ = [[1.0 / n * elem for elem in bounds_i] for bounds_i in bounds]

        # concatenate

        reshape_ = [-1, 1] + list(inputs_[0].shape[1:]) + list(w_out_u.shape[3:])

        bounds_reshape = [
            [K.reshape(elem[0], reshape_), elem[1], K.reshape(elem[2], reshape_), elem[3]] for elem in bounds_
        ]
        w_u_ = K.concatenate([elem[0] for elem in bounds_reshape], 1)
        b_u_ = sum([elem[1] for elem in bounds_reshape], 1)
        w_l_ = K.concatenate([elem[2] for elem in bounds_reshape], 1)
        b_l_ = sum([elem[3] for elem in bounds_reshape])

        bounds_m_0, bounds_m_1 = backward_multiply(
            inputs_0_,
            inputs_1_,
            w_u_,
            b_u_,
            w_l_,
            b_l_,
            convex_domain=self.convex_domain,
            slope=self.slope,
            mode=self.mode,
        )

        shape_0 = [-1, 1] + list(input_0_0[0].shape[1:]) + list(w_u_.shape[3:])
        shape_1 = [-1, 1] + list(input_1_0[0].shape[1:]) + list(w_u_.shape[3:])

        bounds_m_0 = [
            K.reshape(bounds_m_0[0], shape_0),
            bounds_m_0[1],
            K.reshape(bounds_m_0[2], shape_0),
            bounds_m_0[3],
        ]
        bounds_m_1 = [
            K.reshape(bounds_m_1[0], shape_1),
            bounds_m_1[1],
            K.reshape(bounds_m_1[2], shape_1),
            bounds_m_1[3],
        ]

        axes = [i for i in self.axes]
        if axes[0] == -1:
            axes[0] = len(inputs_0[0].shape)
        if axes[1] == -1:
            axes[1] = len(inputs_1[0].shape)

        # import pdb; pdb.set_trace()
        index_0 = np.arange(len(shape_0))
        index_0[2] = axes[0] + 1
        index_0[axes[0] + 1] = 2

        index_1 = np.arange(len(shape_1))
        index_1[2] = axes[1] + 1
        index_1[axes[1] + 1] = 2

        # import pdb; pdb.set_trace()

        bounds_m_0 = [
            K.permute_dimensions(bounds_m_0[0], index_0),
            bounds_m_0[1],
            K.permute_dimensions(bounds_m_0[2], index_0),
            bounds_m_0[3],
        ]
        bounds_m_1 = [
            K.permute_dimensions(bounds_m_1[0], index_1),
            bounds_m_1[1],
            K.permute_dimensions(bounds_m_1[2], index_1),
            bounds_m_1[3],
        ]

        return bounds_m_0, bounds_m_1