from typing import Optional, Any

import keras.ops as K
from keras.layers import Dense


from decomon.layers.layer import DecomonLinearLayer, DecomonLayer
from decomon.types import Tensor
from .utils import Dense_kernel_constraint


class DecomonDense(DecomonLinearLayer):

    def __init__(
        self,
        layer:Dense,
        *args,
        **kwargs: Any,
    ):
        layer_pos = Dense_kernel_constraint(layer=layer, ops=K.maximum, add_bias=True)
        layer_neg = Dense_kernel_constraint(layer=layer, ops=K.minimum, add_bias=False)
        super().__init__(layer=layer, layer_pos=layer_pos, layer_neg=layer_neg, *args,**kwargs)

    def get_affine_representation(self) -> tuple[Tensor, Tensor]:
        w = self.layer.kernel
        b = self.layer.bias if self.layer.use_bias else K.zeros((self.layer.units,))

        # manage tensor-multid input
        for dim in self.layer.input.shape[-2:0:-1]:
            # Construct a multid-tensor diagonal by blocks
            reshaped_outer_shape = (dim, dim) + w.shape
            transposed_outer_axes = (
                (0,)
                + tuple(range(2, 2 + len(b.shape)))
                + (1,)
                + tuple(range(2 + len(b.shape), len(reshaped_outer_shape)))
            )
            w = K.transpose(K.reshape(K.outer(K.identity(dim), w), reshaped_outer_shape), transposed_outer_axes)
            # repeat bias along first dimensions
            b = K.repeat(b[None], dim, axis=0)

        return w, b
