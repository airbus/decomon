from jacobinet.layers.core.activations import BackwardActivation  # type: ignore

from decomon.layers.backward.layer_backward import DecomonNonLinearBackward


class DecomonBackwardActivation(DecomonNonLinearBackward):
    layer: BackwardActivation
