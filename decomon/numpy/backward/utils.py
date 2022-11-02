import numpy as np

from decomon.utils import Grid, Box, Ball
from ..milp.activation import bound_A, bound_B
import tensorflow.keras.backend as K



def get_lower_box(x, w, b):
    """

    :param x: array (n_batch, 2, n_in)
    :param w:  array(n_batch, n_in, n_out)
    :param b: array(n_batch, n_out)
    :return: max_x w*x+ b
    """

    return np.sum(np.maximum(0., w)*x[:, 0, :,None], 1) + np.sum(np.minimum(0., w)*x[:, 1, :,None], 1) + b

def get_upper_box(x, w, b):
    """

    :param x: array (n_batch, 2, n_in)
    :param w:  array(n_batch, n_in, n_out)
    :param b: array(n_batch, n_out)
    :return: max_x w*x+ b
    """
    return np.sum(np.maximum(0., w)*x[:, 1, :,None], 1) + np.sum(np.minimum(0., w)*x[:, 0, :,None], 1) + b


def get_lower(x, w, b, convex_domain={}):
    if not len(convex_domain) or convex_domain['name'] in [Box.name, Grid.name]:
        return get_lower_box(x, w, b)
    else:
        raise NotImplementedError('get_lower is not implemented yet in numpy for the {} domain'.format(convex_domain['name']))


def get_upper(x, w, b, convex_domain={}):
    if not len(convex_domain) or convex_domain['name'] in [Box.name, Grid.name]:
        return get_upper_box(x, w, b)
    else:
        raise NotImplementedError('get_upper is not implemented yet in numpy for the {} domain'.format(convex_domain['name']))


def get_linear_hull_relu(inputs, convex_domain, params=[], **kwargs):

    if len(convex_domain) and convex_domain['name']==Grid.name:
        return get_linear_hull_relu_quantized(inputs, convex_domain=convex_domain, params=params, **kwargs)
    else:
        return get_linear_hull_relu_continuous(inputs, convex_domain=convex_domain, **kwargs)

def get_linear_hull_relu_quantized(inputs,  convex_domain, params=[], **kwargs):

    # compute upper, lower, A, B
    x, w_f_u, b_f_u, w_f_l, b_f_l = inputs[:5]
    # computer upper and lower bounds

    upper = get_upper(x, w_f_u, b_f_u, convex_domain)
    lower = get_lower(x, w_f_l, b_f_l, {})
    relu_u = np.maximum(upper, 0.)
    relu_l = np.maximum(lower, 0.)

    index_unstate = np.maximum(np.sign(upper),0.)*np.maximum(np.sign(-lower),0.)
    index_linear = np.clip(np.sign(lower)+1, 0., 1.)
    index_zero = np.clip(-np.sign(upper)+1, 0., 1.)


    #w_u = (relu_u-relu_l)/np.maximum(upper-lower, 1e-6)
    #b_u = relu_l - w_u*lower

    denum =np.maximum(1e-6, upper-lower)
    w_u = upper/denum
    b_u = - upper*lower/denum


    w_u = (1-index_zero)*w_u
    b_u = (1-index_zero)*b_u
    w_u = (1-index_linear)*w_u + index_linear
    b_u = (1-index_linear)*b_u







    if 'stable_coeff' in convex_domain:
        stable_coeff = convex_domain['stable_coeff']
    else:
        stable_coeff = Grid.stable_coeff
    stable_coeff=0.
    A = compute_A(convex_domain['n'], x, w_f_u, b_f_u, w_f_l, b_f_l, stable_coeff)
    B = compute_B(convex_domain['n'], x, w_f_u, b_f_u, w_f_l, b_f_l, stable_coeff)

    """
    A = np.maximum(A, lower)
    B = np.minimum(B, upper)
    A = np.minimum(A, upper)
    B  = np.maximum(B, lower)
    """
    A =np.minimum(np.minimum(upper, A),0.)
    B  = np.maximum(0., np.maximum(lower, B))

    #A*=0.
    #print('change', len(np.where(info<=0)[0]))
    """
    if len(params):
        A = compute_A(convex_domain['n'], x, w_f_u, b_f_u, w_f_l, b_f_l, stable_coeff)
        B = compute_B(convex_domain['n'], x, w_f_u, b_f_u, w_f_l, b_f_l, stable_coeff)
        A_ = np.minimum(np.max(A, 0), 0.) # use previous params ?
        B_ = np.maximum(np.max(B, 0), 0.)

        if params[0] is None:
            A*=0*A
            B=0.*B
        else:
            #A_ = params[0].numpy()
            #B_ =params[1].numpy()
            K.set_value(params[0], A_)
            K.set_value(params[1], B_)

        if len(params)==3:
            kwargs['slope']=params[2]
        
        #if np.max(np.abs(A_))!=0 or np.max(np.abs(B_))!=0:
        #    import pdb; pdb.set_trace()
    """

    w_l, b_l =get_linear_lower(upper, lower, A, B, **kwargs)


    w_l = (1-index_zero)*w_l
    b_l = (1-index_zero)*b_l
    w_l = (1-index_linear)*w_l + index_linear
    b_l = (1-index_linear)*b_l

    """
    hard_clip = 1- np.sign(np.maximum(upper-lower, 1e-6)/1e-6 -1)
    if np.max(hard_clip):
        print('hello')
        w_l = (1-hard_clip)*w_l + (hard_clip)*K.maximum(0., K.sign(lower))
        w_u = (1 - hard_clip) * w_u + (hard_clip) * K.maximum(0., K.sign(lower))
        b_l = (1 - hard_clip) * b_l
        b_u = (1 - hard_clip) * b_u
    """



    """
    w_l = index_linear + (1-index_linear)*w_l
    b_l = (1-index_linear)*b_l

    # index_unstate
    w_l = (1-index_zero)*w_l
    b_l = (1-index_zero)*b_l
    """

    mask_A = 0 * lower
    mask_B = 0 * upper
    """
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            if A[i, j] - lower[i, j] < 1e-5:
                mask_A[i, j] = 1
            if upper[i, j] - B[i, j] < 1e-5:
                mask_B[i, j] = 1
    """
    mask = mask_A*mask_B

    w_l = (1-mask)*w_l + mask*w_u
    b_l = (1-mask)*b_l + mask*b_u



    w_l = np.concatenate([np.diag(w_l[i])[None] for i in range(len(x))])
    w_u = np.concatenate([np.diag(w_u[i])[None] for i in range(len(x))])
    #import pdb; pdb.set_trace()


    return [w_u, b_u, w_l, b_l]





def compute_A(n, x, w_u, b_u, w_l, b_l, stable_coeff=1e-6):

    # clip A to 0 if A >=-1e-6/2.
    # compute mask

    lower = get_lower(x, w_l, b_l)
    upper = get_upper(x, w_u, b_u)

    mask_A = np.maximum(np.sign(get_lower(x, w_l, b_l)), 0.)+np.maximum(np.sign(-get_upper(x, w_u, b_u)), 0.)
    mask_A = np.minimum(mask_A, 1)
    if np.min(mask_A):
        if np.min(upper*lower)<0:
            import pdb; pdb.set_trace()
        print("all linear")
        return 0*b_u
    A = bound_A(x[:,0], x[:, 1], n, w_u, b_u, w_l, b_l, mask_A)+stable_coeff
    A = np.minimum(A, 0.)

    return A

def compute_B(n, x, w_u, b_u, w_l, b_l, stable_coeff=1e-6):

    #mask_B = np.maximum(np.sign(-get_upper(x, w_u, b_u)), 0.)
    mask_B = np.maximum(np.sign(get_lower(x, w_l, b_l)), 0.) + np.maximum(np.sign(-get_upper(x, w_u, b_u)), 0.)
    mask_B = np.minimum(mask_B, 1)
    if np.min(mask_B):
        #print("all linear")
        return 0.*b_u
    B = bound_B(x[:,0], x[:, 1], n, w_u, b_u, w_l, b_l, mask_B)-stable_coeff
    B = np.maximum(B, 0.)

    return B

def get_linear_lower(upper, lower, A, B, **kwargs):
    #import pdb; pdb.set_trace()
    if 'slope' not in kwargs:
        return get_linear_lower_crown(upper, lower, A, B)
    if kwargs['reuse_slope']:

        index_1 = kwargs['slope'][1].numpy()
        index_2 = kwargs['slope'][2].numpy()

        w_l = 0*upper
        b_l = 0*upper

        denum = np.maximum(B - A, 1e-6)
        w_l_2 = B/denum*np.sign(-A)*np.sign(B)
        b_l_2 = -A*B/denum*np.sign(-A)*np.sign(B)

        w_l += index_1 + index_2*w_l_2
        b_l += index_2*b_l_2

        return [w_l, b_l]


    if kwargs['update_slope']:
        w_l, b_l = get_linear_lower_crown(upper, lower, A, B)


        index_0 = 1-np.sign(w_l)
        index_2 = np.maximum(np.sign((w_l-1)), 0.)
        index_1 = 1-(index_0+index_2)
        alpha_ = np.concatenate([index_0[None], index_1[None], index_2[None]], 0)
        K.set_value(kwargs['slope'], alpha_[:,0])
        return [w_l, b_l]

"""
def get_linear_lower_crown(upper, lower, A, B, w_u, b_u, n, x, eps):


    # function 0: (0, 0)
    x_0 = np.mean(x, 1)

    def func(w, b):
        return (n+1)*b + n*w*np.sum(np.expand_dims(x_0, -1), 1) - eps*np.sum(w)
    error_0 = func(w_u, b_u)
    error_1 = func(w_u-1, b_u)
    denum = np.maximum(B-A, 1e-6)
    error_2 = func(w_u-B/denum, b_u-A*B/denum)

    import pdb; pdb.set_trace()
"""
def get_linear_lower_crown(upper, lower, A, B):

    e_0 = (upper-B)*(upper+B)
    e_1 = -(A-lower)*(A+lower)
    denum = np.maximum(B-A,  1e-6)
    #e_2 = (B*(A-lower)*(B-lower) - A*(upper-B)**2)/denum

    e_2 = (B*(A-lower)**2 - A*(upper-B)**2)/denum
    for i in range(len(lower)):
        for j in range(lower.shape[-1]):
            if A[i,j]*B[i,j]>=0 and upper[i,j]*lower[i,j]<0:
                e_2[i,j] = 10 + max(e_0[i,j], e_1[i,j])


    error_ = np.concatenate([e_2[None], e_1[None], e_0[None]])
    score = np.argmin(error_, 0)

    index_1 = 0 * lower
    index_2 = 0. * lower
    for i in range(len(lower)):
        for j in range(lower.shape[-1]):
            if score[i,j]==1:
                index_1[i,j]=1
            elif score[i,j]==0:
                if upper[i,j]*lower[i,j]<0:
                    print('kikou')
                index_2[i,j]=1

    # if np.max(index_2):
    #    print('youhou')
    w_l = index_1
    w_l += index_2 * B / denum

    b_l = np.zeros_like(upper)
    b_l -= index_2 * (A * B) / denum

    return [w_l, b_l]




def get_linear_lower_crown_old(upper, lower, A, B):

    # print percentage
    # remove degenerate cases
    active = np.where(lower[0]*upper[0]<0)[0]
    N = 1.*np.maximum(len(active), 1)

    rate_A = len(np.where(np.sign(A[0, active])<0))/N
    rate_B = len(np.where(np.sign(B[0, active]) >0))*1./N
    rate_AB = np.sign(B[0, active]*A[0, active])
    rate_AB = len(np.where(rate_AB<0)[0])/N


    print(rate_A*100, rate_B*100, rate_AB*100)
    #print((B-A)/(upper-lower))

    mask_zero = np.sign(np.maximum(upper,0.)) # 1=> u>0
    mask_linear = np.sign(np.maximum(-lower, 0.)) # 1=> l<0

    e_0_ = ((upper-B)*(upper+B))*(mask_zero)
    e_1_ = (-(A-lower)*(A+lower))*mask_linear

    # id
    mask_A = 0 * lower
    mask_B = 0 * upper
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            if A[i, j] - lower[i, j] < 1e-5:
                mask_A[i, j] = 1
            if upper[i, j] - B[i, j] < 1e-5:
                mask_B[i, j] = 1

    denum = np.maximum(B-A, 1e-6)
    e_2_ = np.maximum(B/(denum)*(A-lower)*(B-lower)*(1-mask_A) - A/denum*(upper-B)*(1-mask_B)**2, 0.)


    # penalize e_2_ if upper<=0 or lower>=0
    e_2_ += 10.*np.sign(np.maximum(-upper, 0.)) + 10.*np.sign(np.maximum(lower, 0.))
    # e_2 should be super large if both A and B are 0
    e_2_ = e_2_+ np.maximum(e_0_+1, e_1_+1)*(1-np.abs((np.sign(-A)+np.sign(B)-1)))

    M = np.maximum(e_0_, e_1_, e_2_)
    # if A==lower do not take e_0




    e_0_ += (mask_A )*(M+1)
    # if B==upper do not take e_0
    e_1_ += (mask_B) *(M+1)

    e_0 = e_0_
    e_1 = e_1_
    e_2 = e_2_

    toto = np.maximum(-np.sign(upper*lower), 0.)*e_2



    """
    ttt = np.sign(upper**2-lower**2)*np.sign(e_0-e_1)
    import pdb; pdb.set_trace()

    if np.min(ttt)==-1:
        import pdb;pdb.set_trace()
        ttt+=1.
        ttt = np.minimum(ttt, 1.)
        print('kikouuuuu')
        e_0 = ttt*e_0 + (1-ttt)*upper ** 2
        e_1 = ttt*e_1 + (1-ttt)*lower ** 2
    """
    #e_2 = (1-np.sign(-A*B))*e_2
    error = np.minimum(np.minimum(e_0, e_1), e_2)
    #consider e_2 when A<0 and B>0

    error_ = np.concatenate([e_0[None], e_1[None], e_2[None]])
    score = np.argmin(error_, 0)

    index_1 = 0*lower
    index_2 = 0.*lower
    for i in range(len(lower)):
        for j in range(lower.shape[-1]):
            if np.sign(upper*lower)[i,j]<0 and A[i,j]*B[i,j]<0:
                print('kikou')
                index_2[i,j]=1
            elif score[i,j]==1:
                index_1[i,j]=1
            """
            if score[i,j]==1:
                index_1[i,j]=1
            elif score[i,j]==2:
                print('kikou')
                index_2[i,j]=1
            """
    #if np.max(index_2):
    #    print('youhou')
    w_l = index_1
    w_l += index_2*B/denum

    b_l = np.zeros_like(upper)
    b_l -=index_2*(A*B)/denum


    return [w_l, b_l]





def get_linear_hull_relu_continuous(inputs, convex_domain={}, **kwargs):

    x, w_f_u, b_f_u, w_f_l, b_f_l = inputs[:5]
    # computer upper and lower bounds
    upper = get_upper(x, w_f_u, b_f_u, convex_domain)
    lower = get_lower(x, w_f_l, b_f_l, convex_domain)
    # lower (n_batch, n_out)
    # upper (n_batch, n_out)

    w_l = np.zeros_like(lower)
    b_l = np.zeros_like(lower)

    index_unstate = np.maximum(np.sign(upper),0.)*np.maximum(np.sign(-lower),0.)
    index_linear = np.clip(np.sign(lower)+1, 0., 1.)
    index_zero = np.clip(-np.sign(upper)+1, 0., 1.)

    denum =np.maximum(1e-6, upper-lower)
    w_u = upper/denum
    b_u = - upper*lower/denum

    w_u = (1-index_zero)*w_u
    b_u = (1-index_zero)*b_u
    w_u = (1-index_linear)*w_u + index_linear
    b_u = (1-index_linear)*b_u


    rule = np.maximum(np.sign(upper**2 -lower**2),0.) # to check: identity
    rule*=index_unstate

    w_l += rule
    w_l +=index_linear

    w_u = np.concatenate([np.diag(w_u[i])[None] for i in range(len(x))])
    w_l = np.concatenate([np.diag(w_l[i])[None] for i in range(len(x))])

    return [ w_u, b_u, w_l, b_l]


def merge_with_previous(inputs):
    w_out_u, b_out_u, w_out_l, b_out_l, w_b_u, b_b_u, w_b_l, b_b_l = inputs

    # w_out_u (None, n_h_in, n_h_out)
    # w_b_u (None, n_h_out, n_out)

    # w_out_u_ (None, n_h_in, n_h_out, 1)
    # w_b_u_ (None, 1, n_h_out, n_out)
    # w_out_u_*w_b_u_ (None, n_h_in, n_h_out, n_out)

    # result (None, n_h_in, n_out)

    w_b_u_ = np.expand_dims(w_b_u, 1)
    w_b_l_ = np.expand_dims(w_b_l, 1)
    w_out_u_ = np.expand_dims(w_out_u, -1)
    w_out_l_ = np.expand_dims(w_out_l, -1)
    b_out_u_ = np.expand_dims(b_out_u, -1)
    b_out_l_ = np.expand_dims(b_out_l, -1)

    w_u = np.sum(np.maximum(w_b_u_, 0.)*w_out_u_ + np.minimum(w_b_u_, 0.)*w_out_l_, 2)
    w_l = np.sum(np.maximum(w_b_l_, 0.) * w_out_l_ + np.minimum(w_b_l_, 0.) * w_out_u_, 2)
    b_u = np.sum(np.maximum(w_b_u, 0.)*b_out_u_ + np.minimum(w_b_u, 0.)*b_out_l_, 1) + b_b_u
    b_l = np.sum(np.maximum(w_b_l, 0.) * b_out_l_ + np.minimum(w_b_l, 0.) * b_out_u_, 1) + b_b_l

    return [w_u, b_u, w_l, b_l]







