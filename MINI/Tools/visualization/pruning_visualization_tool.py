import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_prune_ratio(prune_ratios, method, foldername):
    x_axis = []
    y_axis = []
    width = 0.35

    for layer_id in prune_ratios:
        x_axis.append(layer_id)
        y_axis.append(prune_ratios[layer_id])

    ind = np.arange(len(prune_ratios.keys()))
    plt.bar(ind, y_axis, width, label='Input')
    plt.xticks(ind + width / 2, x_axis)
    plt.xlabel('layer id')
    plt.ylabel('percentage')
    plt.title(f"Prune ratio estimation with method {method}")
    plt.legend()
    filename = foldername+f"/prune_ratio_estimation_with_{method}.png"
    plt.ylim((0, 1))
    plt.savefig(filename)
    plt.clf()
    # plt.show()


def plot_criterion_dist(crits):
    filter_sum = []
    filter_g = []
    for layer in crits:
        weights = crits[layer]
        filter_g.append(weights)
        filter_sum = np.concatenate([filter_sum, weights])

    prune_ratio = 0.7
    filters_sum_all = np.sort(np.abs(filter_sum))
    prune_threshold = filters_sum_all[int(0.7*len(filters_sum_all))]
    filter_all_mask = []
    for filters in filter_g:
        filter_group_mask = np.abs(filters) > prune_threshold
        filter_all_mask.append(np.sum(filter_group_mask)/len(filter_group_mask))
    x = np.linspace(0, len(filter_all_mask), len(filter_all_mask))
    plt.bar(x, filter_all_mask)

    plt.title(f"Weight/VGG16/{(prune_ratio)}")
    plt.xlabel("Layer index (Conv2D layer)")
    plt.ylabel("weights creterion")
    plt.show()


