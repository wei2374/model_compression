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


# define nodeType Leaf node, distinguish node, definition of arrow type

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def get_layer_index(dic, layer_target):
    for index, layer in dic.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if layer == layer_target:
            return index


def plotNode(nodeText, centerPt, parentPt, nodeType, ax):
    ax.annotate(nodeText, xy=parentPt, xycoords='data', xytext=centerPt, textcoords='data',
                va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
    # This parameter is a bit scary. did not understand


def plot_sequence(layer, old_end_point, ax, pruned_ratio, layer_index_dic):
    while len(layer.outbound_nodes) == 1 and len(layer.inbound_nodes[0].flat_input_ids) == 1:
        if isinstance(layer, tf.keras.layers.Conv2D) and pruned_ratio is not None:
            layer_index = get_layer_index(layer_index_dic, layer)
            text = f'{layer.name} + prune ratio is {pruned_ratio[layer_index]}'
        else:
            text = f'{layer.name}'
        layer = layer.outbound_nodes[0].layer
        start_point = (old_end_point[0], old_end_point[1]-0.05)
        end_point = (old_end_point[0], old_end_point[1]-0.2)
        plotNode(text, end_point, start_point, leafNode, ax)
        old_end_point = end_point
    if len(layer.outbound_nodes) == 2:
        if isinstance(layer, tf.keras.layers.Conv2D) and pruned_ratio is not None:
            layer_index = get_layer_index(layer_index_dic, layer)
            text = f'{layer.name} + prune ratio is {pruned_ratio[layer_index]}'
        else:
            text = f'{layer.name}'
        start_point = (old_end_point[0], old_end_point[1]-0.05)
        end_point = (old_end_point[0], old_end_point[1]-0.2)
        plotNode(text, end_point, start_point, leafNode, ax)
        old_end_point = end_point

    return layer, old_end_point


def load_model_param(model):
    layer_index_dic = {}
    for index, layer in enumerate(model.layers):
        layer_index_dic[index] = layer

    return layer_index_dic


def visualize_model(model, foldername, pruned_ratio=None):
    ysize = len(model.layers)*90/181
    y_size2 = len(model.layers)*-35/181
    layer_index_dic = load_model_param(model)
    fig, ax = plt.subplots(figsize=(20, ysize))
    ax.set(xlim=(-5, 5), ylim=(y_size2, 3))
    distance = 3
    old_end_point = (0.5, 3)
    layer = model.layers[0]

    while len(layer.outbound_nodes) != 0:
        layer, old_end_point = plot_sequence(
            layer, old_end_point, ax, pruned_ratio, layer_index_dic)

        while len(layer.outbound_nodes) == 2:
            left_start_point = (old_end_point[0]-distance, old_end_point[1])
            right_start_point = (old_end_point[0]+distance, old_end_point[1])
            _, old_end_point = plot_sequence(
                layer.outbound_nodes[0].layer, left_start_point, ax, pruned_ratio, layer_index_dic)
            layer, _ = plot_sequence(
                layer.outbound_nodes[1].layer, right_start_point, ax, pruned_ratio, layer_index_dic)
            old_end_point = (old_end_point[0] + distance, old_end_point[1])

            text = f'{layer.name}'
            start_point = (old_end_point[0], old_end_point[1]-0.05)
            end_point = (old_end_point[0], old_end_point[1]-0.2)
            plotNode(text, end_point, start_point, leafNode, ax)
            old_end_point = end_point

        if len(layer.outbound_nodes) != 0:
            layer = layer.outbound_nodes[0].layer
    plt.savefig(foldername+"/model_plot.png")
    plt.clf()
