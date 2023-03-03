from tensorflow.keras.layers import Wrapper


class BackwardLayer(Wrapper):
    def __init__(self, layer, rec=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.rec = rec

    def get_config(self):
        config = super().get_config()
        config.update({"rec": self.rec})
        return config

    def set_previous(self, previous):
        raise NotImplementedError()
