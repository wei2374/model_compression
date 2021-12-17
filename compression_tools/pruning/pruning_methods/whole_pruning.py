from compression_tools.pruning.delete_filters import delete_filter_after
from compression_tools.pruning.helper_functions import load_model_param
from compression_tools.pruning.pruning_criterions.magnitude import get_and_plot_weights
from compression_tools.pruning.pruning_criterions.activations import get_and_plot_activations
from compression_tools.pruning.pruning_criterions.apoz import get_and_plot_ApoZ
from compression_tools.pruning.pruning_criterions.second_gradient import get_and_plot_gradients2
from compression_tools.pruning.pruning_criterions.first_taylor import get_and_plot_taylor1
from compression_tools.pruning.helper_functions import load_model_param, get_flops_per_channel
import tensorflow as tf
import numpy as np
from tools.visualization.pruning_visualization_tool import plot_prune_ratio
from tools.progress.bar import Bar


def get_filter_to_prune_score(layer_index, layer_params, prun_threshold, score):
    new_layer_param = layer_params[layer_index]
    prune_mask = score < prun_threshold
    prun_neurons = np.where(prune_mask)
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons, num_new_neurons


def prun_filters_score(
            layer_index,
            layer_params,
            prun_threshold,
            layer_index_dic,
            score,
            soft_prune):
    prun_filter, num_new_filter = get_filter_to_prune_score(
                        layer_index,
                        layer_params,
                        prun_threshold,
                        score)
    new_model_param = delete_filter_after(
            layer_params, layer_index, layer_index_dic, prun_filter)

    return new_model_param, num_new_filter, prun_filter


def channel_prune_model_whole(
                my_model,
                prune_ratio,
                foldername=None,
                criterion="gradient",
                get_dataset=None,
                flops_r=1000,
                min_index=1,
                max_index=None,
                soft_prune=False,
                big_kernel_only=False):
    '''
    This function prune the Conv2D layers all together
    '''
    layer_types, layer_params, _, _, layer_index_dic = load_model_param(my_model)
    max_index = len(my_model.layers) if max_index is None else max_index
    # Get the score of channels according to criterions
    if criterion == "magnitude":
        score = get_and_plot_weights(my_model)

    elif criterion == "activation":
        score = get_and_plot_activations(
            my_model, layer_index_dic, get_dataset=get_dataset)

    elif criterion == "gradient1":
        score = get_and_plot_taylor1(
            my_model, layer_index_dic, get_dataset=get_dataset)

    elif criterion == "gradient2":
        score = get_and_plot_gradients2(
            my_model, layer_index_dic, get_dataset=get_dataset)

    elif criterion == "ApoZ":
        score = get_and_plot_ApoZ(
            my_model, layer_index_dic, get_dataset=get_dataset)

    elif criterion == "random":
        score = get_and_plot_weights(my_model)
        from numpy.random import default_rng
        for layer_index in score:
            rng = default_rng()
            new_index = rng.choice(
                len(score[layer_index]), size=int(len(score[layer_index])), replace=False)
            score[layer_index] = score[layer_index][new_index]
    else:
        raise NotImplementedError

    filter_sum = []
    kernel_limit = 1 if big_kernel_only else 0

    # Prepare flops regularization
    flops = get_flops_per_channel(my_model)
    target_layers = {}
    for index, layer in enumerate(my_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) \
                and layer.kernel_size[0] > kernel_limit and\
                index >= min_index and index <= max_index \
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)\
                and prune_ratio != 0 \
                and index in score.keys():
            target_layers[index] = layer

    for index in target_layers:
        if flops_r != 0:
            score[index] = score[index] - (flops[index]/flops_r)
        filter_sum = np.concatenate([filter_sum, score[index]])

    # Get score threshold overall
    filters_sum_all = np.sort((filter_sum))
    prune_threshold = filters_sum_all[int(prune_ratio*len(filters_sum_all))]
    pruned_percentage = {}

    with Bar(f'Whole channel pruning...') as bar:
        for index in target_layers:
            original_channels = target_layers[index].weights[0].shape[-1]
            layer_params, _, _ = prun_filters_score(
                index, layer_params, prune_threshold,
                layer_index_dic, score[index], soft_prune)

            new_channels = layer_params[index][0].shape[-1]
            pruned_percentage[index] = (
               original_channels - new_channels)/float(original_channels)
            # print(f"the channel number for layer(Conv2D){index}\
            #                is reduced {pruned_percentage[index]}")

            bar.next((100/len(target_layers)))
    if foldername is not None:
        plot_prune_ratio(pruned_percentage, criterion, foldername)

    return layer_params, layer_types
