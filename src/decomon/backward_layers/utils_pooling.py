# step1: create the kernel used in max_pooling
# should be a non trainable variable
# compute the toeplitz matrix
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from decomon.backward_layers.utils_conv import get_affine_components
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from decomon.layers.maxpooling import DecomonMaxPooling2D
from decomon.layers.core import ForwardMode
from decomon.core import PerturbationDomain


#(pool_size, pool_size, channels, channels*2**pool_size)
def get_conv_pooling(pool_layer: MaxPooling2D)-> Tuple[tf.Tensor, tf.Tensor]:

    pool_size_x, pool_size_y  = pool_layer.pool_size
    pooling = pool_size_x*pool_size_y
    padding = pool_layer.padding
    strides = pool_layer.strides
    data_format = pool_layer.data_format
    # create the convolution layer to extract the Toeplitz matrix
    kernel_pool = np.repeat(
                    np.transpose(
                        np.eye(pooling).reshape( 
                            (pooling, pool_size_x, pool_size_y)),
                             (1,2, 0))[:,:,None, :], 1, -2)
    op_conv = Conv2D(pooling, (pool_size_x, pool_size_y), 
                        padding=padding, 
                        strides=strides, use_bias=False, data_format=data_format)

    if data_format== "channels_last":
        # should also work with a Decomon layer
        channel_pool = pool_layer.get_input_shape_at(0)
        if isinstance(channel_pool, list):
            channel_pool= channel_pool[-1]
        channel_pool= channel_pool[-1]
        axis_split = -1
    else:
        channel_pool = pool_layer.get_input_shape_at(0)[1]
        if isinstance(channel_pool, list):
            channel_pool= channel_pool[-1]
        channel_pool= channel_pool[1]
        axis_split = 1

    pool_input = pool_layer.get_input_at(0)
    if isinstance(pool_input, list):
        pool_input = pool_input[-1]
    conv_input = tf.split(pool_input, channel_pool, axis=axis_split)[0]
    op_conv(conv_input)
    op_conv.set_weights([kernel_pool])
    op_conv.trainable=False
    op_conv._trainable_weights=[]

    w_conv_u, b_conv_u = get_affine_components(op_conv, [pool_input])
    
    # reshape it given the input
    w_shape = w_conv_u.shape
    dim = int(np.prod(w_shape[2:])/(pooling))
    w_conv_u = K.reshape(tf.repeat(w_conv_u, channel_pool, 2),\
     (-1, pool_size_x, pool_size_y, channel_pool, dim, pooling))
    b_conv_u = K.reshape(b_conv_u, (-1, dim, pooling))
    
    # to check
    return w_conv_u, b_conv_u

def apply_linear_bounds(kernel:tf.Tensor, bias:tf.Tensor, 
                        inputs: List[tf.Tensor], 
                        mode: Union[str, ForwardMode], 
                        perturbation_domain: Optional[PerturbationDomain])-> List[tf.Tensor]:

    dtype = inputs[-1].dtype
    op_dot = K.dot
    z_value = K.cast(0.0, dtype)
    kernel_pos = K.maximum(z_value, kernel)
    kernel_neg = K.minimum(z_value, kernel)

    if mode == ForwardMode.HYBRID:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = inputs
    elif mode == ForwardMode.IBP:
        u_c, l_c = inputs
    elif mode == ForwardMode.AFFINE:
        x_0, w_u, b_u, w_l, b_l = inputs
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode in [ForwardMode.HYBRID, ForwardMode.IBP]:
        #if not self.has_backward_bounds:
        u_c_out = op_dot(u_c, kernel_pos) + op_dot(l_c, kernel_neg)
        l_c_out = op_dot(l_c, kernel_pos) + op_dot(u_c, kernel_neg)

        u_c_out = K.bias_add(u_c_out, bias, data_format="channels_last")
        l_c_out = K.bias_add(l_c_out, bias, data_format="channels_last")

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:

        b_u_out = op_dot(b_u, kernel_pos) + op_dot(b_l, kernel_neg)
        b_l_out = op_dot(b_l, kernel_pos) + op_dot(b_u, kernel_neg)

        b_u_out = K.bias_add(b_u_out, bias, data_format="channels_last")
        b_l_out = K.bias_add(b_l_out, bias, data_format="channels_last")

        w_u_out = op_dot(w_u, kernel_pos) + op_dot(w_l, kernel_neg)
        w_l_out = op_dot(w_l, kernel_pos) + op_dot(w_u, kernel_neg)

    if mode == ForwardMode.HYBRID:
        upper = get_upper(x_0, w_u_out, b_u_out, perturbation_domain)
        lower = get_lower(x_0, w_l_out, b_l_out, perturbation_domain)

        l_c_out = K.maximum(lower, l_c_out)
        u_c_out = K.minimum(upper, u_c_out)

    if mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    if mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    if mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]

    return output

def get_maxpooling_linear_hull(w_conv_pool: tf.Tensor, b_conv_pool: tf.Tensor, 
                                pool_layer: MaxPooling2D, 
                                inputs: List[tf.Tensor], 
                                mode:Union[str, ForwardMode],
                                perturbation_domain: Optional[PerturbationDomain]) -> Tuple[tf.Tensor, tf.Tensor]:

    # forward propagation to bound the second  step of maxpooling, aka maximum
    # we need a routine for applying a linear function
    inputs_max =  apply_linear_bounds(w_conv_pool, b_conv_pool, 
                        inputs, 
                        mode, 
                        perturbation_domain)

    # then derive the linear hull of pooling
    # finetune !!!!
    finetune_lower=None
    finetune_upper=None
    hull_max_lower = get_lower_linear_hull_max(inputs_max, mode, perturbation_domain, axis=-1, 
                                finetune_lower=finetune_lower)
    hull_max_upper = get_upper_linear_hull_max(inputs_max, mode, perturbation_domain, axis=-1, 
                                finetune_upper=finetune_upper)

    # shape issue !!!!

    return merge_with_previous(hull_max_upper+hull_max_lower,\
                               [w_conv_pool, b_conv_pool]*2)

