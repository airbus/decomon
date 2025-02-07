from typing import Any, Optional, Union

from keras.layers import AveragePooling2D
from decomon.layers import DecomonLayer, DecomonLinearLayer
from keras_custom.backward import get_backward


from decomon.types import Tensor
from decomon.layers.utils import get_affine_representation_wo_bias


class DecomonAveragePooling2D(DecomonLinearLayer):
    
    layer: AveragePooling2D
    increasing = True
 