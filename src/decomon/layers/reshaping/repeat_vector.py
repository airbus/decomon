from keras.layers import RepeatVector

from decomon.layers import DecomonLinearLayer


class DecomonRepeatVector(DecomonLinearLayer):
    layer: RepeatVector
    increasing = True
