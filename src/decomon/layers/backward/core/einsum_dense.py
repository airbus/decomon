from jacobinet.layers.core.einsum_dense import BackwardEinsumDense  # type: ignore

from decomon.layers.backward.layer_backward import DecomonLinearLayerBackward


class DecomonBackwardEinsumDense(DecomonLinearLayerBackward):
    layer: BackwardEinsumDense
