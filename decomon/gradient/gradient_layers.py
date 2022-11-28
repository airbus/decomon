from tensorflow.keras.layers import Layer

from decomon.layers.core import F_HYBRID


class GradientDense(Layer):
    """
    Gradient LiRPA of Dense Layer
    """

    def __init__(
        self,
        layer,
        previous=True,
        mode=F_HYBRID.name,
        convex_domain=None,
        finetune=False,
        input_dim=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if convex_domain is None:
            convex_domain = {}
        self.layer = layer

    def call(self, *args, **kwargs):

        return self.kernel
