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
