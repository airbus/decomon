from jacobinet.layers.reshaping.cropping1d import BackwardCropping1D  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardCropping1D(DecomonBackwardLinearLayer):
    layer: BackwardCropping1D
    use_bias = False
    increasing = True
