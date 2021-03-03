from __future__ import absolute_import
from tensorflow.keras.layers import Layer
from abc import ABC, abstractmethod


class StaticVariables:
    """
    Storing static values on the number of input tensors for our layers
    """

    def __init__(self, dc_decomp=False, grad_bounds=False):
        """

        :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
        :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the
        gradient
        """

        # y, z, u_c, w_u, b_u, l_c, w_l, b_l
        nb_tensors = 8

        if dc_decomp:
            nb_tensors += 2
        if grad_bounds:
            # to be determined, set it to upper and lower constants for now
            nb_tensors += 2

        self.nb_tensors = nb_tensors


# create static variables for varying convex domain
class Ball:
    name = "ball"  # Lp Ball around an example


class Box:
    name = "box"  # Hypercube


class Vertex:
    name = "vertex"  # convvex set represented by its vertices
    # (no verification is proceeded to assert whether the set is convex)


class DecomonLayer(ABC, Layer):
    """
    Abstract class that contains the common information of every implemented layers
    """

    def __init__(self, convex_domain={}, dc_decomp=False, grad_bounds=False, first_layer=False, **kwargs):
        """

        :param convex_domain: a dictionary that indicates the type of convex domain we are working on (possible options
        to be determined
        :param dc_decomp: boolean that indicates whether we return a difference of convex decomposition of our layer
        :param grad_bounds: boolean that indicates whether we propagate upper and lower bounds on the values of the
        gradient
        :param first_layer: boolean that indicates whether there are previous layers or not in the computational graph
        :param kwargs:
        """
        super(DecomonLayer, self).__init__(**kwargs)

        self.nb_tensors = StaticVariables(dc_decomp, grad_bounds).nb_tensors
        self.dc_decomp = dc_decomp
        self.grad_bounds = grad_bounds
        self.first_layer = first_layer
        self.convex_domain = convex_domain

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
