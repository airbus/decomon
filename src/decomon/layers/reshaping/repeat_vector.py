from keras.layers import RepeatVector #type:ignore
from decomon.layers import DecomonLinearLayer

class DecomonRepeatVector(DecomonLinearLayer):
    layer: RepeatVector
    increasing = True

