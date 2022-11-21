import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from decomon.utils import get_lower, get_upper


def get_upper_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_u_w, finetune_u_b):
    upper = get_upper(x, W_u - finetune_u_w * W_l, b_u - finetune_u_b * b_l)

    return upper


def get_lower_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_n_w, finetune_n_b):
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

    upper = get_upper_bound_grid_lagrangian(x, W_u, b_u, W_l, b_u, finetune_u_grid_w, finetune_u_grid_b)
    lower = get_lower_bound_grid_lagrangian(x, W_u, b_u, W_l, b_l, finetune_n_grid_w, finetune_n_grid_b)

    return upper, lower
