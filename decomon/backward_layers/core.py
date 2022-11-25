from tensorflow.keras.layers import Layer


class BackwardLayer(Layer):
    def __init__(self, rec=1, **kwargs):
        super().__init__(**kwargs)
        self.rec = rec

    def set_previous(self, previous):
        raise NotImplementedError()
