import numpy as np
import tensorflow as tf
from . import rank_estimator
from pruning.helper_functions import get_flops_per_channel
import torch
from tools.progress.bar import Bar


def get_prune_ratio(
            model,
            param,
            re_method,
            big_kernel_only=False):

    kernel_limit = 1 if big_kernel_only else 0

    prune_ratio = {}
    target_layers = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D)\
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)\
                and layer.kernel_size[0] > kernel_limit:
            target_layers[index] = layer

    with Bar(f'Prune ratio estimation with {re_method} method...') as bar:
        for index in target_layers:
            if re_method == "uniform":
                prune_ratio[index] = param
            elif re_method == "energy":
                cn = rank_estimator.energy_rank(target_layers[index], param=param)
                prune_ratio[index] = 1 - cn
            elif re_method == "VBMF":
                rank = rank_estimator.estimate_ranks_VBMF(
                    target_layers[index], method="channel", parameter=param)
                rank = rank[0]
                output_channel = target_layers[index].output_shape[3]
                ratio = rank/output_channel
                prune_ratio[index] = 1 - ratio
            elif re_method == "BayesOpt":
                rank = rank_estimator.estimate_ranks_BayesOpt(target_layers[index])
                rank = rank[0]
                prune_ratio[index] = 1-(rank/target_layers[index].filters)
                prune_ratio[index] = prune_ratio[index]
            bar.next((100/len(target_layers)))

    return prune_ratio


def estimate_channels_PCA(model, factor=0.5):
    flops = get_flops_per_channel(model)
    whole_score = []
    scores = {}
    channels = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.kernel_size[0] > 1:
            score = []
            layer_data = layer.weights[0]
            dim = layer_data.shape
            channels[index] = dim[3]
            layer_data = np.asarray(layer_data)
            layer_data = layer_data.reshape(dim[0]*dim[1]*dim[2], -1)
            layer_data = torch.tensor(layer_data)
            N, sigmaVH, C = torch.svd(layer_data)
            sum_s = np.sum(np.asarray(sigmaVH))
            pca_score = [s/sum_s for s in np.asarray(sigmaVH)]
            score = [p/flops[index] for p in pca_score]
            scores[index] = score
            for s in score:
                whole_score.append(s)
    whole_score = np.sort(whole_score)
    average_score = whole_score[int(factor*len(whole_score))]
    ranks = {}
    for id in scores:
        mask = scores[id] < average_score
        ranks[id] = np.sum(mask)/channels[id]
    return ranks
