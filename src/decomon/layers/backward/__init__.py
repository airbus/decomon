from jacobinet.layers import (  # non linear layers
    BackwardActivation,
    BackwardAdd,
    BackwardAveragePooling1D,
    BackwardAveragePooling2D,
    BackwardAveragePooling3D,
    BackwardBatchNormalization,
    BackwardConv1D,
    BackwardConv2D,
    BackwardConv3D,
    BackwardCropping1D,
    BackwardCropping2D,
    BackwardCropping3D,
    BackwardDense,
    BackwardDepthwiseConv1D,
    BackwardDepthwiseConv2D,
    BackwardFlatten,
    BackwardGlobalAveragePooling1D,
    BackwardGlobalAveragePooling2D,
    BackwardGlobalAveragePooling3D,
    BackwardPermute,
    BackwardRepeatVector,
    BackwardReshape,
    BackwardUpSampling1D,
    BackwardUpSampling2D,
    BackwardUpSampling3D,
    BackwardZeroPadding1D,
    BackwardZeroPadding2D,
    BackwardZeroPadding3D,
)
from jacobinet.models.utils import FuseGradients, GradConstant
from keras import Layer

from decomon.layers import DecomonLayer
from decomon.layers.backward.convolutional import (
    DecomonBackwardConv1D,
    DecomonBackwardConv2D,
    DecomonBackwardConv3D,
    DecomonBackwardDepthwiseConv1D,
    DecomonBackwardDepthwiseConv2D,
)
from decomon.layers.backward.core import DecomonBackwardActivation, DecomonBackwardDense
from decomon.layers.backward.merging import DecomonBackwardAdd
from decomon.layers.backward.normalization import DecomonBackwardBatchNormalization
from decomon.layers.backward.pooling import (
    DecomonBackwardAveragePooling1D,
    DecomonBackwardAveragePooling2D,
    DecomonBackwardAveragePooling3D,
    DecomonBackwardGlobalAveragePooling1D,
    DecomonBackwardGlobalAveragePooling2D,
    DecomonBackwardGlobalAveragePooling3D,
)
from decomon.layers.backward.reshaping import (
    DecomonBackwardCropping1D,
    DecomonBackwardCropping2D,
    DecomonBackwardCropping3D,
    DecomonBackwardFlatten,
    DecomonBackwardPermute,
    DecomonBackwardRepeatVector,
    DecomonBackwardReshape,
    DecomonBackwardUpSampling1D,
    DecomonBackwardUpSampling2D,
    DecomonBackwardUpSampling3D,
    DecomonBackwardZeroPadding1D,
    DecomonBackwardZeroPadding2D,
    DecomonBackwardZeroPadding3D,
)
from decomon.layers.backward.utils import DecomonFuseGradients, DecomonGradConstant

DECOMON_PREFIX = "Decomon"

default_mapping_jacobinet2decomon_classes: dict[type[Layer], type[DecomonLayer]] = {
    # Add: DecomonAdd,
    # Average: DecomonAverage,
    # Subtract: DecomonSubtract,
    BackwardDense: DecomonBackwardDense,
    # Activation: DecomonActivation,
    # LeakyReLU: DecomonLeakyReLU,
    BackwardConv3D: DecomonBackwardConv3D,
    BackwardConv2D: DecomonBackwardConv2D,
    BackwardConv1D: DecomonBackwardConv1D,
    BackwardDepthwiseConv2D: DecomonBackwardDepthwiseConv2D,
    BackwardDepthwiseConv1D: DecomonBackwardDepthwiseConv1D,
    BackwardZeroPadding1D: DecomonBackwardZeroPadding1D,
    BackwardZeroPadding2D: DecomonBackwardZeroPadding2D,
    BackwardZeroPadding3D: DecomonBackwardZeroPadding3D,
    BackwardCropping1D: DecomonBackwardCropping1D,
    BackwardCropping2D: DecomonBackwardCropping2D,
    BackwardCropping3D: DecomonBackwardCropping3D,
    BackwardFlatten: DecomonBackwardFlatten,
    BackwardRepeatVector: DecomonBackwardRepeatVector,
    BackwardReshape: DecomonBackwardReshape,
    BackwardPermute: DecomonBackwardPermute,
    BackwardUpSampling1D: DecomonBackwardUpSampling1D,
    BackwardUpSampling2D: DecomonBackwardUpSampling2D,
    BackwardUpSampling3D: DecomonBackwardUpSampling3D,
    # BackwardBatchNormalization: DecomonBackwardBatchNormalization, ###
    # BackwardGroupNormalization: DecomonBackwardGroupNormalization, ###
    # BackwardUnitNormalization: DecomonBackwardUnitNormalization, ###
    # BackwardLayerNormalization: DecomonBackwardLayerNormalization, ###
    # BackwardSpectralNormalization: DecomonBackwardSpectralNormalization,
    BackwardAveragePooling1D: DecomonBackwardAveragePooling1D,
    BackwardAveragePooling2D: DecomonBackwardAveragePooling2D,
    BackwardAveragePooling3D: DecomonBackwardAveragePooling3D,
    BackwardGlobalAveragePooling1D: DecomonBackwardGlobalAveragePooling1D,
    BackwardGlobalAveragePooling2D: DecomonBackwardGlobalAveragePooling2D,
    BackwardGlobalAveragePooling3D: DecomonBackwardGlobalAveragePooling3D,
    # Max: DecomonMax,
    # Min: DecomonMin,
    # MaxPooling2D: DecomonMaxPooling2D,
    # MulConstant: DecomonMulConstant,
    # Linear: DecomonLinear,
    # Dropout: DecomonDropout,
    BackwardAdd: DecomonBackwardAdd,
    BackwardBatchNormalization: DecomonBackwardBatchNormalization,
    FuseGradients: DecomonFuseGradients,
    GradConstant: DecomonGradConstant,
    # non linear
    BackwardActivation: DecomonBackwardActivation,
}
