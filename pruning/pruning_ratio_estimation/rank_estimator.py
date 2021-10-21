import tensorly as tl
import tensorflow as tf
import numpy as np
import torch
from . import VBMF
import GPyOpt
from . import BayesOpt
import random
import math


def estimate_rank(layer, rank_selection, param, method):
    if rank_selection == "random":
        ranks = [int(
            random.uniform(0.1, 0.75)*layer.output_shape[3]), int(
                random.uniform(0.2, 0.5)*layer.input_shape[3])]

    elif rank_selection == "VBMF_auto":
        # try:
        ranks = estimate_ranks_VBMF(layer, parameter=None)
        # except Exception:
        #     ranks = estimate_ranks_BayesOpt(layer)

    elif rank_selection == "VBMF":
        # try:
        ranks = estimate_ranks_VBMF(layer, parameter=param)

    elif rank_selection == "weak_VBMF":
        # try:
        ranks = estimate_ranks_VBMF(layer, vbmf_weakenen_factor=0.7)
        # except Exception:
        #     ranks = estimate_ranks_BayesOpt(layer)

    elif rank_selection == "BayesOpt":
        ranks = estimate_ranks_BayesOpt(layer)

    elif rank_selection == "Param":
        ranks = estimate_ranks_param(layer, param, method)

    elif rank_selection == "energy":
        ranks = estimate_ranks_energy(layer, param)

    else:
        raise NotImplementedError

    print(f"{rank_selection} Original output rank is : {layer.output_shape[3]},\
            Original input rank is : {layer.input_shape[3]}")
    print(f"{rank_selection} Estimated output rank is : {ranks[0]},\
            Estimated input rank is : {ranks[1]}")

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


def energy_threshold(sigma, param):
    energy = sigma
    sum_e = torch.sum(energy)
    for i in range(energy.size(0)):
        if energy[:(i+1)].sum()/sum_e >= param:
            valid_idx = i+1
            break

    return valid_idx


def weaken_rank(rank, vbmf_rank, vbmf_weakenen_factor, min_rank=21):
    min_rank = int(min_rank)

    if rank <= min_rank:
        return rank

    if vbmf_rank == 0:
        weaken_rank = rank
    else:
        weaken_rank = int(rank - vbmf_weakenen_factor * (rank - vbmf_rank))
    weaken_rank = max(weaken_rank, min_rank)

    return weaken_rank


def estimate_ranks_energy(layer, param=0.9):
    layer_data = layer.weights[0]
    layer_data = tf.transpose(layer_data, [3, 0, 1, 2])
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    W = layer_data.reshape(dim[0], -1)
    W = torch.tensor(W)
    U, sigma, V = torch.svd(W)
    c_out = energy_threshold(sigma, param)

    layer_data = layer.weights[0]
    layer_data = tf.transpose(layer_data, [2, 0, 1, 3])
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    W = layer_data.reshape(dim[0], -1)
    W = torch.tensor(W)
    U, sigma, V = torch.svd(W)
    c_in = energy_threshold(sigma, param)
    return c_out, c_in


def estimate_ranks_VBMF(
            layer,
            vbmf_weakenen_factor=1,
            min_rank=1,
            method="Tucker",
            parameter=0.00015
        ):
    '''
    VBMF rank estimation

    Key arguments:
    layer -- layer to be decomposed
    vbmf_weakenen_factor -- how much redundancy need to be removed
    min_rank -- if estimated_rank is below min_rank, the min_rank will be used instead
    method -- decompose method

    Return:
    ranks_weaken -- ranks estimated
    '''

    tl.set_backend('tensorflow')

    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)

    if method == "Tucker":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_0 = tl.base.unfold(layer_data, 0)
        unfold_1 = tl.base.unfold(layer_data, 1)

        if parameter is None:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        else:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, parameter)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1, parameter)

        ranks = [diag_0.shape[0], diag_1.shape[1]]
        if vbmf_weakenen_factor == 1:
            return ranks
        else:
            ranks_weaken = [
                weaken_rank(unfold_0.shape[0], ranks[0], vbmf_weakenen_factor, min_rank),
                weaken_rank(unfold_1.shape[0], ranks[1], vbmf_weakenen_factor, min_rank)]
            return ranks_weaken

    if method == "VH":
        layer_data = tf.transpose(layer_data, [2, 0, 3, 1])
        layer_data = np.asarray(layer_data)
        layer_shape = layer_data.shape
        unfold_0 = layer_data.reshape(layer_shape[0]*layer_shape[1], -1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)

        ranks = [diag_0.shape[0]]
        if vbmf_weakenen_factor == 1:
            return ranks

        else:
            ranks_weaken = [
                weaken_rank(unfold_0.shape[0], ranks[0], vbmf_weakenen_factor, min_rank)]
            return ranks_weaken

    if method == "channel":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_0 = tl.base.unfold(layer_data, 0)
        unfold_1 = tl.base.unfold(layer_data, 1)

        if parameter is None:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        else:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, parameter)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1, parameter)

        ranks = [diag_0.shape[0], diag_1.shape[1]]
        if vbmf_weakenen_factor == 1:
            return ranks
        else:
            ranks_weaken = [
                weaken_rank(unfold_0.shape[0], ranks[0], vbmf_weakenen_factor, min_rank),
                weaken_rank(unfold_1.shape[0], ranks[1], vbmf_weakenen_factor, min_rank)]
            return ranks_weaken


def estimate_ranks_param(layer, factor=0.2, method="tucker2D"):
    '''
    Param rank estimation
    Key arguments:
    layer -- layer to be decomposed
    factor -- how many FLOPS in convolutional layer left
    Return:
    max_rank -- ranks estimated
    '''

    tl.set_backend('tensorflow')
    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

    if method == "channel_output" or method == "channel_nl":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*spatial_size*input_channel+output_channel))
        return N, N

    if method == "depthwise":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = math.ceil((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*spatial_size*output_channel+input_channel*output_channel))
        return N, N

    if method == "VH":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*input_channel+spatial_size*output_channel))
        return N, N

    if method == "CP":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*2+output_channel+input_channel))
        return N, N
    # Find max rank for which inequality
    # (initial_count / decomposition_count > rate) holds true
    min_rank = 2
    min_rank = int(min_rank)

    initial_count = np.prod(layer_data.shape)

    cout, cin, kh, kw = layer_data.shape

    beta = max(0.8*(cout/cin), 1.)
    rate = 1./factor

    a = (beta*kh*kw)
    b = (cin + beta*cout)
    c = -initial_count/rate

    discr = b**2 - 4*a*c
    max_rank = int((-b + np.sqrt(discr))/2/a)
    # [R4, R3]
    max1 = layer_data.shape[1]*layer_data.shape[2]*layer_data.shape[3]

    max_rank = max(max_rank, min_rank)
    max_rank = [int(beta*max_rank), max_rank]
    if max_rank[0] > max1:
        max_rank[0] = max1

    max_rank = (max_rank[0], max_rank[1])
    print('Inside estimate, tensor shape: {},\
         max_rank: {}'.format(layer_data.shape, max_rank))
    return max_rank


def estimate_ranks_BayesOpt(layer):
    '''
    BayesOpt rank estimation

    Key arguments:
    layer -- layer to be decomposed

    Return:
    ranks -- ranks estimated
    '''

    func = BayesOpt.BayesOpt_rank_selection(layer)

    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

    axis_0 = layer_data.shape[0]
    axis_1 = layer_data.shape[1]

    space = [
        {"name": "rank_1", "type": "continuous", "domain": (2, axis_0 - 1)},
        {"name": "rank_2", "type": "continuous", "domain": (2, axis_1 - 1)},
    ]

    feasible_region = GPyOpt.Design_space(space=space)

    initial_design = GPyOpt.experiment_design.initial_design(
        "random", feasible_region, 10
    )

    objective = GPyOpt.core.task.SingleObjective(func.f)

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(
        model, feasible_region, optimizer=acquisition_optimizer
    )

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(
        model, feasible_region, objective, acquisition, evaluator, initial_design
    )

    max_time = None
    tolerance = 10e-3
    max_iter = 10
    bo.run_optimization(
        max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True
    )

    # bo.plot_acquisition()
    # bo.plot_convergence()

    rank1 = int(bo.x_opt[0])
    rank2 = int(bo.x_opt[1])
    ranks = [rank1, rank2]

    return ranks


# def estimate_ranks_PCA(model, factor=0.7):
#     flops = get_flops_per_channel(model)
#     whole_score = []
#     scores = {}
#     for index, layer in enumerate(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1:
#             score = []
#             layer_data = layer.weights[0]
#             dim = layer_data.shape
#             layer_data = np.asarray(layer_data)
#             layer_data = layer_data.reshape(dim[0]*dim[1]*dim[2], -1)
#             layer_data = torch.tensor(layer_data)
#             N, sigmaVH, C = torch.svd(layer_data)
#             sum_s = np.sum(np.asarray(sigmaVH))
#             pca_score = [s/sum_s for s in np.asarray(sigmaVH)]
#             score = [p/flops[index] for p in pca_score]
#             scores[index] = score
#             for s in score:
#                 whole_score.append(s)
#     whole_score = np.sort(whole_score)
#     average_score = whole_score[int(factor*len(whole_score))]
#     ranks = {}
#     for id in scores:
#         mask = scores[id] > average_score
#         ranks[id] = np.sum(mask)
#     return ranks
