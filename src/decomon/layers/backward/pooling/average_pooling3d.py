from jacobinet.layers.pooling.average_pooling3d import (
    BackwardAveragePooling3D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardAveragePooling3D(DecomonLinearLayerBackward):
    layer: BackwardAveragePooling3D
    use_bias = False
    increasing = True
