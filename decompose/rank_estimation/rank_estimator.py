import tensorflow as tf
import random
from .rank_estimate_bayesopt import estimate_ranks_BayesOpt
from .rank_estimate_energy import estimate_ranks_energy
from .rank_estimate_param import estimate_ranks_param
from .rank_estimate_vbmf import estimate_ranks_VBMF


def estimate_rank(layer, rank_selection, param, schema):
    if rank_selection == "random":
        ranks = [int(
            random.uniform(0.1, 0.75)*layer.output_shape[3]), int(
                random.uniform(0.2, 0.5)*layer.input_shape[3])]

    elif rank_selection == "VBMF_auto":
        ranks = estimate_ranks_VBMF(layer, noise_variance=None, schema=schema)

    elif rank_selection == "VBMF":
        ranks = estimate_ranks_VBMF(layer, noise_variance=param, schema=schema)

    elif rank_selection == "BayesOpt":
        ranks = estimate_ranks_BayesOpt(layer)

    elif rank_selection == "Param":
        ranks = estimate_ranks_param(layer, param, schema)

    elif rank_selection == "energy":
        ranks = estimate_ranks_energy(layer, param)

    else:
        raise NotImplementedError

    # print(f"{rank_selection} Original output rank is : {layer.output_shape[3]},\
    #         Original input rank is : {layer.input_shape[3]}")
    # print(f"{rank_selection} Estimated output rank is : {ranks[0]},\
    #         Estimated input rank is : {ranks[1]}")

    return ranks


def get_flops_per_channel(model):
    flops = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
            C_in = layer.get_weights()[0].shape[2]
            C_out = layer.get_weights()[0].shape[3]
            Kernel = layer.get_weights()[0].shape[0] * layer.get_weights()[0].shape[1]
            H = layer.input_shape[1]
            W = layer.input_shape[2]
            FLOPS_1 = 2*H*W*C_in*C_out*Kernel/C_out

            i = index+1
            FLOPS_2 = 0
            while(i < len(model.layers)):
                if isinstance(model.layers[i], tf.compat.v1.keras.layers.Conv2D):
                    layer2 = model.layers[i]
                    C_in = layer2.get_weights()[0].shape[2]
                    C_out = layer2.get_weights()[0].shape[3]
                    Kernel = layer2.get_weights()[0].shape[1] * layer2.get_weights()[0].shape[0]
                    H = layer2.input_shape[1]
                    W = layer2.input_shape[2]
                    FLOPS_2 = 2*H*W*C_in*C_out*Kernel/C_in
                    break
                else:
                    i += 1

            flops[index] = float(FLOPS_1 + FLOPS_2)/(10 ** 6)
    return flops
