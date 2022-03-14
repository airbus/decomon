from __future__ import absolute_import
from ..layers.utils import exp, log, expand_dims, add, minus, sum
from decomon.layers.utils import F_HYBRID


# compute categorical cross entropy

def categorical_cross_entropy(input_, dc_decomp=False, mode=F_HYBRID.name, convex_domain={}):

    # step 1: exponential
    exp_ = exp(input_, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)
    # step 2: sum
    sum_exp_ = sum(exp_, axis=-1, mode=mode)
    # step 3
    tmp = log(sum_exp_, dc_decomp=dc_decomp, mode=mode, convex_domain=convex_domain)

    log_sum_exp = expand_dims(tmp,
                              dc_decomp=dc_decomp,
                              mode=mode, convex_domain=convex_domain, axis=-1)


    log_p = add(minus(input_, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp),
                log_sum_exp, mode=mode, convex_domain=convex_domain, dc_decomp=dc_decomp)

    return log_p
