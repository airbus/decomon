from keras.layers import Reshape #type:ignore
from decomon.layers import DecomonLinearLayer
import keras.ops as K #type:ignore


class DecomonReshape(DecomonLinearLayer):
    layer: Reshape
    increasing= True
    
