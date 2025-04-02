from keras.layers import AveragePooling1D #type:ignore
from decomon.layers import DecomonLinearLayer


class DecomonAveragePooling1D(DecomonLinearLayer):
    
    layer: AveragePooling1D
    linear= True
    increasing = True
