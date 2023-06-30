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
from decomon.layers.utils_pooling import get_lower_linear_hull_max, get_upper_linear_hull_max
from decomon.backward_layers.utils import merge_with_previous

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

    pooling_shape = pool_input.get_shape().as_list()[1:]
    w_shape = w_conv_u.get_shape().as_list()

    dim = int(np.prod(w_shape[1:])*channel_pool/(pooling*np.prod(pooling_shape)))
    w_conv_u = tf.repeat(w_conv_u, channel_pool, 2)
    w_conv_u = tf.repeat(K.reshape(w_conv_u, [-1]+pooling_shape+[dim, pooling]), channel_pool, -2)
    b_conv_u = tf.repeat(K.reshape(b_conv_u, (-1, dim, pooling)), channel_pool, -2)

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

        # expand u_c, l_c
        axis_sum = len(u_c.shape)
        u_c = K.expand_dims(K.expand_dims(u_c, -1), -1)
        l_c = K.expand_dims(K.expand_dims(l_c, -1), -1)
        u_c_out = K.sum(u_c*kernel_pos + l_c*kernel_neg, np.arange(1, axis_sum))
        l_c_out = K.sum(l_c*kernel_pos + u_c*kernel_neg, np.arange(1, axis_sum))
        
        u_c_out += bias
        l_c_out += bias
        #u_c_out = K.bias_add(u_c_out, bias, data_format="channels_last")
        #l_c_out = K.bias_add(l_c_out, bias, data_format="channels_last")

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        axis_sum = len(b_u.shape)
        b_u = K.expand_dims(K.expand_dims(b_u, -1), -1)
        b_l = K.expand_dims(K.expand_dims(b_l, -1), -1)
        
        b_u_out = K.sum(b_u*kernel_pos + b_l*kernel_neg, np.arange(1, axis_sum))
        b_l_out = K.sum(b_l*kernel_pos + b_u*kernel_neg, np.arange(1, axis_sum))
        
        b_u_out += bias
        b_l_out += bias
        
        w_u = K.expand_dims(K.expand_dims(w_u, -1), -1)
        w_l = K.expand_dims(K.expand_dims(w_l, -1), -1)
        kernel_pos_w = K.expand_dims(kernel_pos, 1)
        kernel_neg_w = K.expand_dims(kernel_neg, 1)

        w_u_out = K.sum(w_u*kernel_pos_w + w_l*kernel_neg_w, np.arange(2, axis_sum+1))
        w_l_out = K.sum(w_l*kernel_pos_w + w_u*kernel_neg_w, np.arange(2, axis_sum+1))

    # not necessary anymore with Nolwen's trick
    """
    if mode == ForwardMode.HYBRID:
        upper = get_upper(x_0, w_u_out, b_u_out, perturbation_domain)
        lower = get_lower(x_0, w_l_out, b_l_out, perturbation_domain)

        l_c_out = K.maximum(lower, l_c_out)
        u_c_out = K.minimum(upper, u_c_out)
    """

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
    w_hull_l, b_hull_l = get_lower_linear_hull_max(inputs_max, mode, perturbation_domain, axis=-1, 
                                finetune_lower=finetune_lower)

    w_hull_u, b_hull_u = get_upper_linear_hull_max(inputs_max, mode, perturbation_domain, axis=-1, 
                                finetune_upper=finetune_upper)
    shape_w = w_hull_l.get_shape().as_list()[1:]
    shape_prod = np.prod(shape_w)
    w_hull_l = K.sum(K.reshape(tf.linalg.diag(K.reshape(w_hull_l, (-1, shape_prod))), [-1, shape_prod]+shape_w), -1)
    w_hull_u = K.sum(K.reshape(tf.linalg.diag(K.reshape(w_hull_u, (-1, shape_prod))), [-1, shape_prod]+shape_w), -1)

    # shape issue !!!!
    # flatten w_conv_pool
    dim_conv = w_conv_pool.get_shape().as_list()[1:]
    dim_input_conv = int(np.prod(dim_conv)/np.prod(dim_conv[-2:]))
    dim_output_conv = np.prod(dim_conv[-2:])
    w_conv_flat = K.reshape(w_conv_pool, (-1, dim_input_conv, dim_output_conv))
    b_conv_flat = K.reshape(b_conv_pool, (-1, dim_output_conv))
    
    return merge_with_previous([w_conv_flat, b_conv_flat]*2+[w_hull_u, b_hull_u]+[w_hull_l, b_hull_l])

