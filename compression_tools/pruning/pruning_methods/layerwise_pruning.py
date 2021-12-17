from compression_tools.pruning.new_delete import delete_filter_after
from compression_tools.pruning.helper_functions import load_model_param
from compression_tools.pruning.pruning_criterions.magnitude import get_and_plot_weights
from compression_tools.pruning.pruning_criterions.activations import get_and_plot_activations
from compression_tools.pruning.pruning_criterions.apoz import get_and_plot_ApoZ
from compression_tools.pruning.pruning_criterions.second_gradient import get_and_plot_gradients2
from compression_tools.pruning.pruning_criterions.first_taylor import get_and_plot_taylor1
from compression_tools.pruning.pruning_criterions.batch_norm import get_and_plot_BN
from compression_tools.pruning.helper_functions import load_model_param
import tensorflow as tf
import numpy as np
from tools.progress.bar import Bar


def get_filter_to_prune_layer(layer_index, layer_params, prun_ratio, score):
    new_layer_param = layer_params[layer_index]
    score_sorted = np.sort(score)
    index = int(prun_ratio*len(score_sorted))
    prun_threshold = score_sorted[index]
    prune_mask = score <= prun_threshold
    prun_neurons = np.where(prune_mask)
    num_new_neurons = new_layer_param[0].shape[-1] - len(prun_neurons)
    return prun_neurons, num_new_neurons


def get_all_filters(score, prune_ratio):
    filters = {}
    for sc in score:
        if sc in prune_ratio:
            score_sorted = np.sort(score[sc])
            index = int(prune_ratio[sc]*len(score_sorted))
            prun_threshold = score_sorted[index]
            prune_mask = score[sc] <= prun_threshold
            filter = np.where(prune_mask)
        filters[sc]=filter
    return filters

def prun_filters_layer(
            layer_index,
            layer_params,
            prun_ratio,
            layer_index_dic,
            score,
            soft_prune,
            layer_type="conv2D"):

    # Get filters to be pruned
    prun_filter, num_new_filter = get_filter_to_prune_layer(
        layer_index, layer_params, prun_ratio, score)

    # Prune the model params
    new_model_param = delete_filter_after(
        layer_params, layer_index, layer_index_dic,
        prun_filter, soft_prune=soft_prune, layer_type=layer_type)

    return new_model_param, num_new_filter, prun_filter


def channel_prune_model_layerwise(
        my_model,
        prune_ratio,
        criterion="gradient1",
        get_dataset=None,
        min_index=1,
        max_index=None,
        soft_prune=False,
        big_kernel_only=True,
        option="CL"):
    '''Prune each Conv2D layer separately
    Args:
        my_model : (keras model) the model to be pruned
        prune_ratio : (float list) how many channels is to be pruned in each channel
        method : (str) channel selection criterion
        dataset: (str) dataset
        min_index : (int) start prunning from this layer
        soft_prune : (bool) whether or not use soft_prune
        big_kernel_only : (bool) whether or not prune conv layers with small kernel
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

    elif criterion == "BN":
        score = get_and_plot_BN(
             my_model, layer_index_dic)

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

    kernel_limit = 1 if big_kernel_only else 0

    # Start pruning each layer
    filters = {}
    target_layers = {}
    for index, layer in enumerate(my_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) \
                and layer.kernel_size[0] > kernel_limit \
                and (option == "CL" or option == "CLFL") and\
                index >= min_index and index <= max_index \
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)\
                and prune_ratio[index] != 0 \
                and index in score.keys():
            target_layers[index] = layer

    filters = get_all_filters(score, prune_ratio)
    # Prune the model params
    with Bar(f'Layerwise channel pruning...') as bar:
        for index in target_layers:
            new_model_param = delete_filter_after(
                layer_params, index, layer_index_dic,
                filters, soft_prune=soft_prune)
            bar.next((100/len(target_layers)))
        
    return new_model_param, layer_types