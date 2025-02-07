from keras.layers import Reshape
from decomon.layers import DecomonLinearLayer
import keras.ops as K


class DecomonReshape(DecomonLinearLayer):
    layer: Reshape
    increasing= True
