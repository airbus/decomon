# step1: create the kernel used in max_pooling
# should be a non trainable variable
# compute the toeplitz matrix
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MaxPooling2D
from decomon.backward_layers.utils_conv import get_affine_components, get_toeplitz
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Input
from decomon.layers.maxpooling import DecomonMaxPooling2D
from decomon.layers.core import ForwardMode
from decomon.core import PerturbationDomain
from decomon.layers.utils_pooling import get_lower_linear_hull_max, get_upper_linear_hull_max
from decomon.backward_layers.utils import merge_with_previous

def get_pool_info_at_build(config_pool:Dict, input_shape:Tuple[int])->Dict:
    """Generate the config dictionary of the inner convolution inside a MaxPooling

    Args:
        config_pool: config dict of a pool layer
        input_shape: input shape of the pool layer

    Returns:
        config dict of the inner convolution
    """

    pool_size_x, pool_size_y = config_pool['pool_size']
    padding = config_pool['padding']
    strides = config_pool['strides']
    data_format = config_pool['data_format']

    if data_format== "channels_last":
        channel_pool= input_shape[-1]
        axis_split = -1
    else:
        channel_pool = input_shape[1]
        channel_pool= channel_pool[1]
        axis_split = 1

    config={'data_format':data_format,
    'pool_size_x':pool_size_x,
    'pool_size_y': pool_size_y,
    'padding':padding,
    'strides':strides,
    'channel_pool':channel_pool,
    'axis':axis_split}

    return config

def get_pool_info(pool_layer: MaxPooling2D)->Dict:
    """Generate the config dictionary of the inner convolution inside a MaxPooling

    Args:
        pool_layer: pool layer used for Lirpa

    Returns:
        config dict of the inner convolution
    """
    return get_pool_info_at_build(pool_layer.get_config(), pool_layer.get_input_shape_at(0))

def get_conv_op_pooling(pool_config, input_shape)-> Tuple[tf.Tensor, Conv2D]:

    config = get_pool_info_at_build(pool_config, input_shape)
    pool_size_x = config['pool_size_x']
    pool_size_y = config['pool_size_y']
    pooling = pool_size_x*pool_size_y
    padding = config['padding']
    strides = config['strides']
    data_format = config['data_format']
    channel_pool = config['channel_pool']
    axis_split = config['axis']

    if padding=='same':
        raise NotImplementedError()
    # create the convolution layer to extract the Toeplitz matrix
    kernel_pool = np.repeat(
                    np.transpose(
                        np.eye(pooling).reshape( 
                            (pooling, pool_size_x, pool_size_y)),
                             (1,2, 0))[:,:,None, :], 1, -2)
    op_conv = Conv2D(pooling, (pool_size_x, pool_size_y), 
                        padding=padding, 
                        strides=strides, use_bias=False, data_format=data_format)


    pool_input = Input(input_shape[1:])
    conv_input = tf.split(pool_input, channel_pool, axis=axis_split)[0]
    op_conv(conv_input)
    op_conv.set_weights([kernel_pool])
    op_conv.trainable=False
    op_conv._trainable_weights=[]

    return pool_input, op_conv

def get_conv_pooling(pool_config:Dict, input_shape:Tuple[int])-> Tuple[tf.Tensor, tf.Tensor]:
    """Generate the inner convolution inside a MaxPooling

    Args:
        pool_config: config dict of a pool layer
        input_shape: input shape of the pool layer

    Returns:
        Conv2D such that pooling= Sequential[conv, max]
    """

    config = get_pool_info_at_build(pool_config, input_shape)
    pool_size_x = config['pool_size_x']
    pool_size_y = config['pool_size_y']
    pooling = pool_size_x*pool_size_y
    channel_pool = config['channel_pool']
    pool_input, op_conv = get_conv_op_pooling(pool_config, input_shape)

    w_conv_u = get_toeplitz(op_conv)[None]
    pooling_shape = pool_input.get_shape().as_list()[1:]
    w_shape = w_conv_u.get_shape().as_list()

    #check
    dim = int(np.prod(w_shape[1:])*channel_pool/(pooling*np.prod(pooling_shape)))
    
    w_conv_u = tf.linalg.diag(tf.repeat(K.expand_dims(w_conv_u, -1), channel_pool, -1))
    w_conv_u = K.permute_dimensions(w_conv_u, (0,1, 3, 2, 4))
    w_conv_u = K.reshape(w_conv_u, [-1]+pooling_shape+[dim, pooling, channel_pool])
    w_conv_u = K.reshape(K.permute_dimensions(w_conv_u, (0, 1, 2, 3, 4, 6, 5)),\
                         [-1]+pooling_shape+[dim*channel_pool, pooling])
    
    return tf.constant(w_conv_u.numpy(), dtype=w_conv_u.dtype)


def apply_linear_bounds(kernel:tf.Tensor, bias:Union[None,tf.Tensor], 
                        inputs: List[tf.Tensor], 
                        mode: Union[str, ForwardMode], 
                        perturbation_domain: PerturbationDomain)-> List[tf.Tensor]:
    """Private routine to propagate lirpa bounds after a linear layer. 
    Used for maxpooling that is cut in two sequential operations: conv + max

    Args:
        kernel: kernel of the affine operator
        bias: bias of the affine operator
        inputs: lirpa inputs of the affine operator
        mode: type of Forward propagation (ibp, affine, or hybrid). Default to hybrid.
        perturbation_domain: type of perturbation domain that encompass the set of perturbations. Defaults to None.

    Raises:
        ValueError: Unknown mode

    Returns:
        output tensors after the affine operator.
    """

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
        if not(bias is None): 
            u_c_out += bias
            l_c_out += bias

    if mode in [ForwardMode.HYBRID, ForwardMode.AFFINE]:
        axis_sum = len(b_u.shape)
        b_u = K.expand_dims(K.expand_dims(b_u, -1), -1)
        b_l = K.expand_dims(K.expand_dims(b_l, -1), -1)
        
        b_u_out = K.sum(b_u*kernel_pos + b_l*kernel_neg, np.arange(1, axis_sum))
        b_l_out = K.sum(b_l*kernel_pos + b_u*kernel_neg, np.arange(1, axis_sum))
        if not(bias is None):
            b_u_out += bias
            b_l_out += bias
        
        w_u = K.expand_dims(K.expand_dims(w_u, -1), -1)
        w_l = K.expand_dims(K.expand_dims(w_l, -1), -1)
        kernel_pos_w = K.expand_dims(kernel_pos, 1)
        kernel_neg_w = K.expand_dims(kernel_neg, 1)

        w_u_out = K.sum(w_u*kernel_pos_w + w_l*kernel_neg_w, np.arange(2, axis_sum+1))
        w_l_out = K.sum(w_l*kernel_pos_w + w_u*kernel_neg_w, np.arange(2, axis_sum+1))

    if mode == ForwardMode.HYBRID:
        output = [x_0, u_c_out, w_u_out, b_u_out, l_c_out, w_l_out, b_l_out]
    if mode == ForwardMode.AFFINE:
        output = [x_0, w_u_out, b_u_out, w_l_out, b_l_out]
    if mode == ForwardMode.IBP:
        output = [u_c_out, l_c_out]

    return output

def get_maxpooling_linear_hull(w_conv_pool: tf.Tensor, 
                                inputs: List[tf.Tensor], 
                                mode:Union[str, ForwardMode],
                                perturbation_domain: PerturbationDomain) -> Tuple[tf.Tensor, tf.Tensor]:
    """Linear hull of maxpooling. The maxpooling is based on two primitives operator: a convolution and a max

    Args:
        w_conv_pool: kernel of the inner convolution
        inputs: lirpa inputs
        mode: type of Forward propagation (ibp, affine, or hybrid). Default to hybrid.
        perturbation_domain: type of perturbation domain that encompass the set of perturbations. Defaults to None.


    Returns:
        output tensors
    """

    # forward propagation to bound the second  step of maxpooling, aka maximum
    # we need a routine for applying a linear function
    inputs_max =  apply_linear_bounds(w_conv_pool, None,
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
    
    #cannot use merge_with_previous because the batch dimension is not the same
    #it creates a ValueError at runtime
    # do merge_with_previous by hand
    # as the affine components are positive, it reduces the number of operations

    w_pool_u = K.sum(K.expand_dims(w_conv_flat, -1)*K.expand_dims(w_hull_u, 1), -2)
    w_pool_l = K.sum(K.expand_dims(w_conv_flat, -1)*K.expand_dims(w_hull_l, 1), -2)

    return [w_pool_u, b_hull_u, w_pool_l, b_hull_l]

