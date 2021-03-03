from __future__ import absolute_import
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np

# from decomon.layers.core import DecomonLayer
from ..layers.core import DecomonLayer
from ..layers.decomon_layers import to_monotonic
from ..layers.utils import softmax_to_linear
from tensorflow.python.keras.engine.base_layer import InputSpec
from ..models.decomon_sequential import clone_to_monotonic, convert_to_monotonic

VGG_DIM = 224 * 224 * 3


class PreprocessVGG(Layer):
    """"""

    def __init__(self, **kwargs):
        super(PreprocessVGG, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def call(self, input):
        variable = K.variable(np.array([123.68 / 255.0, 116.779 / 255.0, 103.939 / 255.0]))
        # if self.from_pixel:
        #    # input_ = K.tf.floor(input)
        #    # input_ = K.floor(input)
        #    output = input - variable[None, None, None]
        # else:
        #    output = input - variable[None, None, None]
        output = input - variable[None, None, None]
        return output[:, :, :, ::-1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(PreprocessVGG, self).get_config()
        return config


class DecomonPreprocessVGG(PreprocessVGG, DecomonLayer):
    """"""

    def __init__(self, **kwargs):
        super(DecomonPreprocessVGG, self).__init__(**kwargs)
        self.input_spec = [
            InputSpec(ndim=4),
            InputSpec(ndim=4),
            InputSpec(ndim=2),
            InputSpec(ndim=2),
            InputSpec(ndim=4),
            InputSpec(ndim=5),
            InputSpec(ndim=4),
            InputSpec(ndim=4),
            InputSpec(ndim=5),
            InputSpec(ndim=4),
        ]

    def call(self, inputs):

        h, g, x_min, x_max, u_c, w_u, b_u, l_c, w_l, b_l = inputs
        call_super = super(DecomonPreprocessVGG, self).call

        h_ = call_super(h)
        b_u_ = call_super(b_u)
        b_l_ = call_super(b_l)
        u_c_ = call_super(u_c)
        l_c_ = call_super(l_c)
        g_ = g[:, :, :, ::-1]
        w_u_ = w_u[:, :, :, :, ::-1]
        w_l_ = w_l[:, :, :, :, ::-1]

        return [h_, g_, x_min, x_max, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    def compute_output_shape(self, input_shape):
        return input_shape


def to_monotonic_vgg(layer, input_dim=VGG_DIM):
    """

    :param layer:
    :return:
    """

    if isinstance(layer, PreprocessVGG):
        return DecomonPreprocessVGG.from_config(layer.get_config())

    return to_monotonic(layer, input_dim)


def convert_to_monotonic_VGG16(normalized_input=True, input_tensors=None, input_dim=VGG_DIM, *args, **kwargs):
    """

    :param normalized_input:
    :param input_tensors:
    :param args:
    :param kwargs:
    :return:
    """

    # load vgg
    vgg_model = vgg16.VGG16(*args, **kwargs)
    vgg_model = softmax_to_linear(vgg_model)

    layers = vgg_model.layers

    for layer in layers:
        print(np.prod(layer.get_output_shape_at(0)[1:]))
    # vgg_model = Model(vgg_model.input, keras.layers.Flatten()(layers[:2]))
    vgg_model = Sequential(layers=layers[:2] + [keras.layers.Flatten()])

    for layer in vgg_model.layers:
        print(layer.get_output_shape_at(0))

    if normalized_input:
        vgg_model_ = Sequential(layers=[PreprocessVGG(), vgg_model])
        vgg_model_(vgg_model.input)
        vgg_model = vgg_model_

    return vgg_model, convert_to_monotonic(
        vgg_model, input_tensors=None, layer_fn=to_monotonic_vgg, input_dim=input_dim
    )
