from tensorflow.keras.layers import Wrapper


class BackwardLayer(Wrapper):
    def __init__(self, layer, rec=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.rec = rec

    def set_previous(self, previous):
        raise NotImplementedError()
