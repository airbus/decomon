# step1: create the kernel used in max_pooling
# should be a non trainable variable
# compute the toeplitz matrix

import numpy as np

#(pool_size, pool_size, channels, channels*2**pool_size)
def get_conv_pooling(pool_layer: MaxPooling2D)-> tf.Tensor:

    pool_size_x, pool_size_y  = pool_layer.pool_size
    pooling = pool_size_x*pool_size_y
    padding = pool_layer.padding
    strides = pool_layer.strides
    # create the convolution layer to extract the Toeplitz matrix
    kernel_pool = np.repeat(
                    np.transpose(
                        np.eye(pooling).reshape( 
                            (pooling**2, pool_size_x, pool_size_y)),
                             (1,2, 0))[:,:,None, :], 1, -2)
    
    op_conv = Conv2D(pooling, (pool_size_x, pool_size_y), 
                        padding=padding, 
                        strides=strides, use_bias=False)

    op_conv(pool_layer.get_input_at(0))
    op_conv.set_weights([kernel_pool])
    op_conv.trainable=False
    op_conv._trainable_weights=[]

    # retrieve the linear hull
    



def get_kernel_pooling(pool_size, channels):

    # expand dim/broadcast
    mask = tf.linalg.diag(tf.ones((pool_size), dtype=dtype))  # (n_dim, n_dim)
    mask_shape = np.ones(len(u_c.shape) + 1)
    mask_shape[-1] = n_dim
    if axis != -1:
        mask_shape[axis] = n_dim
    else:
        mask_shape[-2] = n_dim
    mask = K.reshape(mask, mask_shape)  # (1, shape, n_dim, n_dim)