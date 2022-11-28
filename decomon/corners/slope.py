import tensorflow as tf
from tensorflow.python.keras import backend as K


def get_linear_lower_slope_relu(upper, lower, upper_g_: tf.Variable, lower_g_: tf.Variable, **kwargs):

    # test the nature of A and B (variable or tensor)
    upper_g = 0.0 * upper + upper_g_
    lower_g = 0.0 * upper + lower_g_

    A = K.minimum(upper_g, 0.0)
    B = K.maximum(lower_g, 0.0)

    mask_zero = K.sign(K.maximum(upper, 0.0))
    mask_linear = K.sign(K.maximum(-lower, 0.0))

    e_0_ = ((upper - B) * (upper + B)) * mask_zero  # 0
    e_1_ = (-(A - lower) * (A + lower)) * mask_linear

    # id
    denum = K.maximum(B - A, 1e-6)
    e_2_ = K.maximum(B / denum * (A - lower) * (B - lower) - A / denum * (upper - B) ** 2, 0.0)

    # penalize e_2_ if upper<=0 or lower>=0
    e_2_ += 10.0 * (1 - mask_linear) + 10 * (1 - mask_zero)
    # e_2 should be super large if both A and B are 0

    M = e_0_ + e_1_ + e_2_
    # if A==lower do not take e_0
    e_0_ += (1 - K.sign(A - lower)) * (1 - mask_linear) * (M + 1)
    # if B==upper do not take e_0
    e_1_ += (1 - K.sign(upper - B)) * (1 - mask_zero) * (M + 1)
    e_2_ = e_2_ + M * K.abs((K.sign(-A) + K.sign(B) - 1))

    e_0 = e_0_
    e_1 = e_1_
    e_2 = e_2_

    error = K.minimum(K.minimum(e_0, e_1), e_2)
    index_1 = 1 - K.sign(e_1 - error)  # identity
    index_2 = 1 - K.sign(e_2 - error)

    w_l = index_1
    w_l += index_2 * B / denum

    b_l = K.zeros_like(upper)
    b_l -= index_2 * (A * B) / denum

    if "finetune" in kwargs:

        alpha = kwargs["finetune"]
        alpha_0 = alpha[0][None]
        alpha_1 = alpha[1][None]
        alpha_2 = K.minimum(alpha[2][None], (1 - alpha_1))

        # condition by A<0 and B>0
        w_l_2 = alpha_2 * B / denum
        b_l_2 = -alpha_2 * (A * B) / denum

        w_l_1 = alpha[1][None] + 0 * w_l
        b_l_1 = 0 * b_l

        w_l_bis = w_l_1 + w_l_2
        b_l_bis = b_l_1 + b_l_2

        w_l = alpha_0 * w_l + (1 - alpha_0) * w_l_bis
        b_l = alpha_0 * b_l + (1 - alpha_0) * b_l_bis

    return [w_l, b_l]
