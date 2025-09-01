from keras.layers import GroupNormalization

from decomon.layers import DecomonLayer


class DecomonGroupNormalization(DecomonLayer):
    layer: GroupNormalization
    diagonal = True
