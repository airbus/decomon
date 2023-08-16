from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import backward_max_
from decomon.backward_layers.utils_pooling import get_conv_pooling, get_maxpooling_linear_hull

from decomon.core import PerturbationDomain
from decomon.layers.core import ForwardMode
#from decomon.utils import get_lower, get_upper


class BackwardMaxPooling2D(BackwardLayer):
    """Backward  LiRPA of MaxPooling2D"""

    def __init__(
        self,
        layer: Layer,
        rec: int = 1,
        mode: Union[str, ForwardMode] = ForwardMode.HYBRID,
        perturbation_domain: Optional[PerturbationDomain] = None,
        dc_decomp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            layer=layer,
            rec=rec,
            mode=mode,
            perturbation_domain=perturbation_domain,
            dc_decomp=dc_decomp,
            **kwargs,
        )
        pool_config = self.layer.get_config()
        input_shape = self.layer.get_input_shape_at(0)
        if isinstance(input_shape, list):
            input_shape=input_shape[-1]
        w_conv = get_conv_pooling(pool_config, input_shape)
        self.w_conv = w_conv # add non trainable


    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        return get_maxpooling_linear_hull(w_conv_pool=self.w_conv, 
                                inputs=inputs, 
                                mode=self.mode,
                                perturbation_domain=self.perturbation_domain)
