from contextlib import closing

import NNet  # https://github.com/sisl/NNet
import numpy as np
from NNet.utils.readNNet import readNNet


def convert_nnet_2_numpy(repo, filename, clipping=True, normalize_in=True, normalize_out=False, verbose=0):
    with closing(open(f"{repo}/{filename}", "rb")) as f:
        lines = f.readlines()

    index = 0
    while lines[index].decode("utf-8")[:2] == "//":
        index += 1

    # get network architecture
    archi = [int(e) for e in lines[index + 1].decode("utf-8").strip().split(",")[:-1]]
    input_dim = archi[0]
    n_layers = archi[1:]
    if verbose:
        print("archi", archi)

    # get MIN and MAX of the domain
    MIN = lines[index + 3].decode("utf-8").strip().split(",")[:-1]
    MAX = lines[index + 4].decode("utf-8").strip().split(",")[:-1]

    # get normalization
    MEAN = lines[index + 5].decode("utf-8").strip().split(",")[:-1]
    STD = lines[index + 6].decode("utf-8").strip().split(",")[:-1]
    MIN = np.array([float(e) for e in MIN])
    MAX = np.array([float(e) for e in MAX])
    MEAN = np.array([float(e) for e in MEAN])
    STD = np.array([float(e) for e in STD])
    # print('MEAN', MEAN)
    # print('STD', STD)

    # split into normalization of input and output
    STD_out = STD[-1]
    STD_in = STD[:-1]
    MEAN_out = MEAN[-1]
    MEAN_in = MEAN[:-1]

    # retrieve the parameters of the networks (weights and biases)
    weights, biases = readNNet(f"{repo}/{filename}")
    params = []
    for w, b in zip(weights, biases):
        params += [w.T, b]

    if normalize_in:
        W = np.diag(1.0 / STD_in)
        bias = -MEAN_in / STD_in

        w_0, b_0 = params[:2]
        w_0_ = np.dot(w_0.T, W).T
        b_0_ = np.dot(w_0.T, bias) + b_0
        params[0] = w_0_
        params[1] = b_0_

    if normalize_out:
        w_1, b_1 = params[-2:]
        w_1 /= STD_out
        b_1 = (b_1 - MEAN_out) / STD_out
        params[-2] = w_1
        params[-1] = b_1

    # create the equivalent numpy function
    def func(x, clip=clipping):
        n_dim = 2
        if len(x.shape) == 1:
            x = x[None]
            n_dim = 1

        if clip:
            x = np.maximum(x, MIN[None])
            x = np.minimum(x, MAX[None])

        w_ = params[::2]
        b_ = params[1::2]

        for i in range(len(w_) - 1):
            x = np.dot(x, w_[i]) + b_[i][None]
            x = np.maximum(x, 0.0)
        x = np.dot(x, w_[-1]) + b_[-1][None]
        if n_dim == 1:
            return x[0]
        else:
            return x

    return func
