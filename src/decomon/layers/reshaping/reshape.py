from keras.layers import Reshape
from decomon.layers import DecomonLayer, DecomonLinearLayer
import keras.ops as K

from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias

class DecomonReshape(DecomonLinearLayer):
    layer: Reshape
    increasing= True
