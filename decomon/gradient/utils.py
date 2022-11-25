# compute gradient of function based on bounds on its input
from tensorflow.python.keras import backend as K

from decomon.layers import F_FORWARD, F_HYBRID, F_IBP, StaticVariables
from decomon.layers.utils import get_lower, get_upper


def gradient_relu(inputs, dc_decomp=False, mode=F_HYBRID.name, convex_domain=None, **kwargs):

    # get upper and lower
    if convex_domain is None:
        convex_domain = {}
    if mode not in [F_HYBRID.name, F_IBP.name, F_FORWARD.name]:
        raise ValueError(f"unknown mode {mode}")

    nb_tensors = StaticVariables(dc_decomp=False, mode=mode).nb_tensors
    if mode == F_HYBRID.name:
        x_0, u_c, w_u, b_u, l_c, w_l, b_l = x[:nb_tensors]
    elif mode == F_IBP.name:
        u_c, l_c = x[:nb_tensors]
    elif mode == F_FORWARD.name:
        x_0, w_u, b_u, w_l, b_l = x[:nb_tensors]
    else:
        raise ValueError(f"Unknown mode {mode}")

    if mode == F_FORWARD.name:
        upper = get_upper(x_0, w_u, b_u, convex_domain)
        lower = get_lower(x_0, w_l, b_l, convex_domain)
    if mode in [F_IBP.name, F_HYBRID.name]:
        upper = u_c
        lower = l_c

    # gradient is K.sign(x)
    if mode in [F_IBP.name, F_HYBRID.name]:
        u_c_ = K.sign(upper)
        l_c_ = K.sign(lower)

    if mode in [F_FORWARD.name, F_HYBRID.name]:

        w_u_, b_u_, w_l_, b_l_ = get_linear_hull_sign(upper, lower, **kwargs)
        b_u_ = w_u_ * b_u + b_u_
        b_l_ = w_l_ * b_l + b_l_
        w_u_ = K.expand_dims(w_u_, 1) * w_u
        w_l_ = K.expand_dims(w_l_, 1) * w_l

    output = []
    if mode == F_IBP.name:
        output += [u_c_, l_c_]
    if mode == F_FORWARD.name:
        output += [x_0, w_u_, b_u_, w_l_, b_l_]
    if mode == F_HYBRID.name:
        output += [x_0, u_c_, w_u_, b_u_, l_c_, w_l_, b_l_]

    return output


def get_linear_hull_sign(upper, lower, **kwargs):

    z_value = K.cast(0.0, K.floatx())
    o_value = K.cast(1.0, K.floatx())

    # ones all the time
    active = K.clip(K.sign(lower) + o_value, z_value, o_value)  # active =1 if lower >=0
    # zero all the time
    inactive = -K.clip(K.sign(upper) - o_value, -o_value, z_value)  # inactive =1 if upper<=0

    # else either w_u = 0, b_u=1 or (w_u*0+b_u=1, w_u*lower+b_u=0) => b_u=1 and w_u = -1/lower

    # assuming lower<=0
    # w_u=0, b_u=1: |lower|
    # w_u= 1/|lower|: 0.5*|lower| + 0.5*upper*(1+1/|lower|*upper -1)
    # ...    0.5*|lower| + 0.5*upper*2/|lower|

    # condition for the first case: |lower|**2<= upper**2

    condition_bool = K.maximum(K.sign(upper**2 - K.abs(lower) ** 2), z_value)  # 1 then w_u=0, b_u=1

    w_u_ = (o_value - condition_bool) / K.maximum(K.abs(lower), K.epsilon())
    b_u_ = condition_bool + (o_value - condition_bool) * lower

    # w_l=0, b_l=1
    # b_l=0 w_u = 1/upper

    w_l_ = condition_bool / K.maximum(K.abs(upper), K.epsilon())
    b_l_ = o_value - condition_bool

    # apply linear conditions
    w_u_ = (o_value - active) * w_u_
    b_u_ = (o_value - active) * b_u_ + active
    w_l_ = (o_value - active) * w_l_
    b_l_ = (o_value - active) * b_l_ + active

    # dead conditions
    w_u_ = (o_value - inactive) * w_u_
    b_u_ = (o_value - inactive) * b_u_
    w_l_ = (o_value - inactive) * w_l_
    b_l_ = (o_value - inactive) * b_l_

    return [w_u_, b_u_, w_l_, b_l_]
