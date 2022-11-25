from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from ..layers.utils import get_lower, get_upper


def get_lower_bound_grid(x, W, b, n):

    A, B = convert_lower_search_2_subset_sum(x, W, b, n)
    return subset_sum_lower(A, B, repeat=n)


def get_upper_bound_grid(x, W, b, n):

    return -get_lower_bound_grid(x, -W, -b, n)


def get_bound_grid(x, W_u, b_u, W_l, b_l, n):

    upper = get_upper_bound_grid(x, W_u, b_u, n)
    lower = get_lower_bound_grid(x, W_l, b_l, n)

    return upper, lower


# convert max Wx +b s.t Wx+b<=0 into a subset-sum problem with positive values
def convert_lower_search_2_subset_sum(x, W, b, n):

    x_min = x[:, 0]
    x_max = x[:, 1]

    const = get_lower(x, W, b, convex_domain={})

    if len(W.shape) > 3:
        W = K.reshape(W, (-1, W.shape[1], np.prod(W.shape[2:])))

    weights = K.abs(W) * K.expand_dims((x_max - x_min) / n, -1)
    return weights, const


def subset_sum_lower(W, b, repeat=1):

    B = tf.sort(W, 1)
    C = K.repeat_elements(B, rep=repeat, axis=1)
    C_ = K.cumsum(C, axis=1)

    if len(b.shape) > 2:
        shape = np.prod(b.shape[1:])
        b_ = K.reshape(b, (-1, shape))
    else:
        b_ = b

    D = K.minimum(K.sign(K.expand_dims(-b_, 1) - C_) + 1, 1)

    score = K.minimum(2 * K.sum(D * C, 1) + b_, 0.0)
    return score


def get_upper_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_u_w, finetune_u_b):

    # do better with masks
    upper = get_upper(x, W_u - finetune_u_w * W_l, b_u - finetune_u_b * b_l)

    return upper


def get_lower_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_n_w, finetune_n_b):
    # do better with masks
    lower = get_lower(x, W_l + finetune_n_w * W_u, b_l + finetune_n_b * b_u)

    return lower


def get_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_grid):

    finetune_n_grid_w, finetune_u_grid_w = finetune_grid
    finetune_n_grid_b, finetune_u_grid_b = finetune_grid

    # add batch dimension
    # add input dimension
    finetune_u_grid_b = finetune_u_grid_b[None]
    finetune_u_grid_w = finetune_u_grid_w[None, None]
    finetune_n_grid_b = finetune_n_grid_b[None]
    finetune_n_grid_w = finetune_n_grid_w[None, None]

    # clip weights !!!
    upper = get_upper_bound_grid_lagrangian(x, W_u, b_u, W_l, b_u, finetune_u_grid_w, finetune_u_grid_b)
    lower = get_lower_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_n_grid_w, finetune_n_grid_b)

    return upper, lower
