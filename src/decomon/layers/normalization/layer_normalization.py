from keras.layers import LayerNormalization

from decomon.layers import DecomonLayer


class DecomonLayerNormalization(DecomonLayer):
    layer: LayerNormalization
    diagonal = True
