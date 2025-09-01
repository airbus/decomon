from keras_custom.layers import MulConstant

from decomon.layers.layer import DecomonLayer


class DecomonMulConstant(DecomonLayer):
    layer: MulConstant
    linear = True
    diagonal = True
    use_bias = False
