from decomon.layers import get_lower, get_upper
from tensorflow.python.keras import backend as K


def backward_relu_(x, w_out_u, b_out_u, w_out_l, b_out_l, convex_domain={}):

    # get input bounds
    _, x_0, u_c, w_u, b_u, l_c, w_l, b_l = x

    # compute upper and lower bounds
    # and keep the minimum between the constant and the computed upper bounds
    # and keep the maximum between the constant and the computer lower bounds
    upper = K.minimum(u_c, get_upper(x_0, w_u, b_u, convex_domain))
    lower = K.maximum(l_c, get_lower(x_0, w_l, b_l, convex_domain))

    # check inactive relu state: u<=0
    index_dead = -K.clip(K.sign(upper), -1, 0)
    index_linear = K.clip(K.sign(lower) + 1, 0, 1)

    # 1 if upper<=-lower else 0
    index_a = -K.clip(K.sign(upper + lower) - 1, -1.0, 0.0)
    # 1 if upper>-lower else 0
    index_b = K.ones_like(index_a) - index_a

    # in case upper=lower, this cases are
    # considered with index_dead and index_linear
    w_u_0 = upper / K.maximum(K.epsilon(), upper - lower)
    b_u_0 = -upper * lower / K.maximum(K.epsilon(), upper - lower)

    w_u_ = (
        index_dead * K.zeros_like(w_u_0) + index_linear * K.ones_like(w_u_0) + (1.0 - index_dead - index_linear) * w_u_0
    )
    b_u_ = (index_dead + index_linear) * K.zeros_like(b_u_0) + (1 - index_linear - index_dead) * b_u_0

    # lower bound
    b_l_ = K.zeros_like(b_u_)
    w_l_ = K.zeros_like(w_u_)
    w_l_ += K.maximum(index_b, index_linear) * K.ones_like(w_l_)

    return w_u_, b_u_, w_l_, b_l_


def backward_linear_prod(x_0, bounds_x, back_bounds, convex_domain):
    """

    :param bounds_x:
    :param back_bounds:
    :return:
    """
    w_u_i, b_u_i, w_l_i, b_l_i = bounds_x
    w_u, b_u, w_l, b_l = back_bounds

    if len(w_u_i.shape) > 3:
        n_dim = w_u_i.get_input_shape_at(0)[1]
        w_u_i = K.reshape(w_u_i, (-1, n_dim, n_dim))
        w_l_i = K.reshape(w_l_i, (-1, n_dim, n_dim))
        b_u_i = K.reshape(b_u_i, (-1, n_dim))
        b_l_i = K.reshape(b_l_i, (-1, n_dim))

    # reshape bounds_x !!!!

    x_max = get_upper(x_0, w_u_i - w_l_i, b_u_i - b_l_i, convex_domain)
    mask_b = 1.0 - K.sign(x_max)
    mask_a = 1.0 - mask_b

    w_u_i_ = K.expand_dims(K.expand_dims(w_u_i, 1), -1)
    w_l_i_ = K.expand_dims(K.expand_dims(w_l_i, 1), -1)
    b_u_i_ = K.expand_dims(K.expand_dims(b_u_i, 1), -1)
    b_l_i_ = K.expand_dims(K.expand_dims(b_l_i, 1), -1)
    mask_a = K.expand_dims(K.expand_dims(mask_a, 1), -1)
    mask_b = K.expand_dims(K.expand_dims(mask_b, 1), -1)

    w_u_pos = K.maximum(w_u, 0.0)
    w_u_neg = K.minimum(w_u, 0.0)
    w_l_pos = K.maximum(w_l, 0.0)
    w_l_neg = K.minimum(w_l, 0.0)

    w_u_pos_ = K.expand_dims(w_u_pos, 2)
    w_u_neg_ = K.expand_dims(w_u_neg, 2)
    w_l_pos_ = K.expand_dims(w_l_pos, 2)
    w_l_neg_ = K.expand_dims(w_l_neg, 2)
    mask_a_ = K.expand_dims(mask_a, 2)
    mask_b_ = K.expand_dims(mask_b, 2)

    w_u_ = K.sum(mask_a_ * (w_u_pos_ * w_u_i_ + w_u_neg_ * w_l_i_), 3) + K.sum(
        K.expand_dims(w_u, 2) * mask_b_ * w_u_i_, 3
    )
    w_l_ = K.sum(mask_a_ * (w_l_pos_ * w_l_i_ + w_l_neg_ * w_u_i_), 3) + K.sum(
        K.expand_dims(w_l, 2) * mask_b_ * w_l_i_, 3
    )

    b_u_ = K.sum(mask_a * (w_u_pos * b_u_i_ + w_u_neg * b_l_i_), 2) + K.sum(mask_b * (w_u * b_u_i_), 2) + b_u
    b_l_ = K.sum(mask_a * (w_l_pos * b_l_i_ + w_l_neg * b_u_i_), 2) + K.sum(mask_b * (w_l * b_l_i_), 2) + b_l

    return [w_u_, b_u_, w_l_, b_l_]
