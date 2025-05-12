from jacobinet.layers.core.einsum_dense import BackwardEinsumDense  # type: ignore

from decomon.layers.backward.layer_backward import DecomonBackwardLinearLayer


class DecomonBackwardEinsumDense(DecomonBackwardLinearLayer):
    layer: BackwardEinsumDense
