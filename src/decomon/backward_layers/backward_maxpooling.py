from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Flatten, Layer

from decomon.backward_layers.core import BackwardLayer
from decomon.backward_layers.utils import backward_max_
from decomon.backward_layers.utils_pooling import get_conv_pooling

from decomon.core import PerturbationDomain
from decomon.layers.core import ForwardMode
from decomon.utils import get_lower, get_upper


class BackwardMaxPooling2D(BackwardLayer):
    """Backward  LiRPA of MaxPooling2D"""

    pool_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: str
    data_format: str
    fast: bool

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
        w_conv, b_conv = get_conv_pooling(self.layer)
        self.w_conv = w_conv
        self.b_conv = b_conv


    def call(self, inputs: List[tf.Tensor], **kwargs: Any) -> List[tf.Tensor]:

        return get_maxpooling_linear_hull(w_conv_pool=self.w_conv, 
                                          b_conv_pool=self.b_conv, 
                                pool_layer=self.layer, 
                                inputs=inputs, 
                                mode=self.mode,
                                perturbation_domain=self.perturbation_domain)

        return self._pooling_function(
            inputs=inputs_wo_backward_bounds,
            w_u_out=w_u_out,
            b_u_out=b_u_out,
            w_l_out=w_l_out,
            b_l_out=b_l_out,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            perturbation_domain=self.perturbation_domain,
        )
