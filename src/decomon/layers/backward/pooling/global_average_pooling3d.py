from jacobinet.layers.pooling.global_average_pooling3d import (
    BackwardGlobalAveragePooling3D,  # type: ignore
)

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardGlobalAveragePooling3D(DecomonBackwardLinearLayer):
    layer: BackwardGlobalAveragePooling3D
    use_bias = False
    increasing = True
