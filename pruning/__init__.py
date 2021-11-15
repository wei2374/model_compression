import tensorflow as tf
from tools.visualization.model_visualization import visualize_model
from .pruning_ratio_estimation.channel_estimator import get_prune_ratio
from .pruning_methods.layerwise_pruning import channel_prune_model_layerwise
from .pruning_methods.whole_pruning import channel_prune_model_whole
from .pruning_methods.lasso_pruning import channel_prune_model_lasso
from tools.visualization.pruning_visualization_tool import plot_prune_ratio


def build_pruned_model(original_model, new_model_param, layer_types, foldername):
    model_config = original_model.get_config()
    prune_ratio = {}
    for i in range(0, len(model_config['layers'])):
        if model_config['layers'][i]['class_name'] == "Dense":
            model_config['layers'][i]['config']['units'] = new_model_param[i][0].shape[1]

        elif model_config['layers'][i]['class_name'] == "Conv2D":
            channel_output = model_config['layers'][i]['config']['filters']
            prune_ratio[i] = 1 - new_model_param[i][0].shape[3]/channel_output
            model_config['layers'][i]['config']['filters'] = new_model_param[i][0].shape[3]
            if (model_config['layers'][i+2]['class_name']) == "ReLU":
                model_config['layers'][i]['config']['trainable'] = True


    original_model = tf.keras.Model().from_config(model_config)
    original_model.build(input_shape=original_model.input_shape)

    for i in range(0, len(original_model.layers)):
        if layer_types[i] == 'Conv2D' or\
           layer_types[i] == 'Dense'\
           or layer_types[i] == 'BatchNormalization' or\
           layer_types[i] == 'DepthwiseConv2D' and i > 0:
            original_model.layers[i].set_weights(new_model_param[i])
    if foldername is not None:
        visualize_model(original_model, foldername, prune_ratio)
    return original_model


def model_prune(
        raw_model,
        dataset,
        method,
        re_method,
        param,
        criterion,
        max_index,
        min_index=1,
        foldername=None,
        big_kernel_only=False,
        option="CL",
        ):
    '''
    Prune layers based on pruning settings

    Key arguments:
    raw_model -- model to be pruned
    foldername -- folder used to save related logging and pictures
    method -- pruning method
    schema -- decomposition schemes
    re_method -- Methods for estimating the prune ratio for each layer
    param -- parameter necessary for prune ratio estimation methods
    criterion -- channel importance estimation methods
    min_index -- only prune the layers whose id is behind min_index
    max_index -- only prune the layers whose id is before min_index
    
    big_kernel_only -- if only prune layer with large receptive field
    option -- "CL"-> only prune convolutional layers
              "FL"-> only prune fully connected layers
              "CLFL"-> prune convolutional layers as well as
                     fully connected layers

    Return:
    pruned_model -- model that is pruned
    '''
    if method == "layerwise":
        prune_ratio = get_prune_ratio(
                    raw_model,
                    param,
                    re_method=re_method,
                    big_kernel_only=big_kernel_only)
        if foldername is not None:
            plot_prune_ratio(prune_ratio, re_method, foldername)

        layer_params, layer_types = channel_prune_model_layerwise(
                    raw_model,
                    prune_ratio,
                    criterion=criterion,
                    dataset=dataset,
                    option=option,
                    min_index=min_index,
                    max_index=max_index,
                    big_kernel_only=big_kernel_only
                )

        pruned_model = build_pruned_model(raw_model, layer_params, layer_types, foldername)
        return pruned_model

    if method == "whole":
        layer_params, layer_types = channel_prune_model_whole(
                    raw_model,
                    param,
                    flops_r=0,
                    foldername=foldername,
                    criterion=criterion,
                    dataset=dataset,
                    min_index=min_index,
                    max_index=max_index,
                    big_kernel_only=big_kernel_only)
        pruned_model = build_pruned_model(raw_model, layer_params, layer_types, foldername)
        return pruned_model

    if method == "lasso":
        prune_ratio = get_prune_ratio(
                    raw_model,
                    param,
                    re_method=re_method,
                    big_kernel_only=big_kernel_only)
        if foldername is not None:
            plot_prune_ratio(prune_ratio, re_method, foldername)

        layer_params, layer_types = channel_prune_model_lasso(
                    raw_model,
                    prune_ratio,
                    min_index=min_index,
                    max_index=max_index,
                    dataset=dataset,)
        pruned_model = build_pruned_model(raw_model, layer_params, layer_types, foldername)
        return pruned_model
