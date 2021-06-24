from __future__ import absolute_import
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod


class F_IBP:
    name = "ibp"


class F_FORWARD:
    name = "forward"


class F_HYBRID:
    name = "hybrid"


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Vertex:
    name = "vertex"  # convvex set represented by its vertices
    # (no verification is proceeded to assert whether the set is convex)


class StaticVariables:
    """
    Storing static values on the number of input tensors for our layers
    """

    def __init__(self, dc_decomp=False, grad_bounds=False, mode=F_HYBRID.name):
        """

        :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
        :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the
        gradient
        """

        self.mode = mode

        if self.mode == F_HYBRID.name:
            # y, z, u_c, w_u, b_u, l_c, w_l, b_l
            nb_tensors = 8
        elif self.mode == F_IBP.name:
            # y, z, u_c, l_c
            nb_tensors = 4
        elif self.mode == F_FORWARD.name:
            # y, z, w_u, b_u, w_l, b_l
            nb_tensors = 6
        else:
            raise NotImplementedError("unknown forward mode {}".format(mode))

        if dc_decomp:
            nb_tensors += 2
        if grad_bounds:
            # to be determined, set it to upper and lower constants for now
            nb_tensors += 2

        self.nb_tensors = nb_tensors


class DecomonLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers
    """

    def __init__(
        self, convex_domain={}, dc_decomp=False, mode=F_HYBRID.name, grad_bounds=False, fast=True, n_subgrad=0, **kwargs
    ):
        """

        :param convex_domain: a dictionary that indicates the type of convex
        domain we are working on (possible options
        to be determined
        :param dc_decomp: boolean that indicates whether we return a
        difference of convex decomposition of our layer
        :param grad_bounds: boolean that indicates whether we propagate upper
        and lower bounds on the values of the
        gradient
        :param n_subgrad: integer that indicates the number of
        subgradient descent steps for linear layers
        :param mode: ...
        :param kwargs:
        """
        super(DecomonLayer, self).__init__(**kwargs)

        self.nb_tensors = StaticVariables(dc_decomp, grad_bounds, mode).nb_tensors
        self.dc_decomp = dc_decomp
        self.grad_bounds = grad_bounds
        self.convex_domain = convex_domain
        self.n_subgrad = n_subgrad
        self.mode = mode
        self.fast = fast

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
