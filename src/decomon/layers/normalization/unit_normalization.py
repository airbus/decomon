from keras.layers import UnitNormalization

from decomon.layers import DecomonLayer


class DecomonUnitNormalization(DecomonLayer):
    layer: UnitNormalization
    diagonal = True
