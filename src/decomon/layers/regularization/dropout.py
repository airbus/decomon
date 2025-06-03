# onnx should use a custom library of keras...
from keras.layers import Dropout

from decomon.layers import DecomonLayer
from decomon.layers.utils.affine import get_affine_representation_wo_bias
from decomon.types import Tensor


class DecomonDropout(DecomonLayer):
    layer: Dropout
    linear = True
    diagonal = True
    use_bias = False
