import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_channel_numbers(model):
    '''
    get the input and output channels of all layers in a model
    '''
    layer_channels = {}
    layer_inputs = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D) and layer.kernel_size[0] > 1:
            layer_channels[index] = layer.filters
            layer_inputs[index] = layer.weights[0].shape[2]

    return layer_channels, layer_inputs


def get_and_plot_gamma_distribution(model, prune_ratio=0.7, dataset='cifat10'):
    original_model = model
    gamma_dic = {}

    # Sort gammas and find threshold
    for index, layer in enumerate(original_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma_dic[index] = layer.gamma
    gamma_list = list(gamma_dic.values())
    gamma_sum = []
    for gamma_group in gamma_list:
        gamma_sum = np.concatenate([gamma_sum, np.asarray(gamma_group)])
    gamma_n = len(gamma_sum)
    gamma_sum = np.sort(np.abs(gamma_sum))
    gamma_threshold = gamma_sum[int(gamma_n*prune_ratio)]

    # Get percentage of big gamma in each layer
    gamma_mask_all = []
    for gamma_group in gamma_list:
        gamma_mask_group = np.abs(gamma_group) > gamma_threshold
        gamma_mask_all.append(gamma_mask_group)
    gamma_p = []
    for gamma_mask_group in gamma_mask_all:
        gamma_p.append(np.sum(gamma_mask_group)/len(gamma_mask_group))

    # get weights from selected channels
    model_params = {}
    counter = 0
    first_d = True
    for index, layer in enumerate(original_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            conv_layer = original_model.layers[index-1]
            weights = conv_layer.get_weights()
            if counter == 0:
                weights[0] = weights[0][:, :, :, gamma_mask_all[counter]]
            else:
                weights[0] = weights[0][:, :, :, gamma_mask_all[counter]]
                weights[0] = weights[0][:, :, gamma_mask_all[counter-1], :]
            weights[1] = weights[1][gamma_mask_all[counter]]
            model_params[index-1] = weights

            bn_layer = original_model.layers[index]
            weights = bn_layer.get_weights()
            weights[0] = weights[0][gamma_mask_all[counter]]
            weights[1] = weights[1][gamma_mask_all[counter]]
            weights[2] = weights[2][gamma_mask_all[counter]]
            weights[3] = weights[3][gamma_mask_all[counter]]
            model_params[index] = weights
            counter += 1
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = original_model.layers[index]
            weights = dense_layer.get_weights()
            if first_d:
                weights[0] = weights[0][gamma_mask_all[counter-1], :]
                first_d = False
            model_params[index] = weights

    x = range(len(gamma_p))
    plt.bar(x, gamma_p)
    plt.title(f"BN gamma/VGG16/{prune_ratio}/{dataset}")
    plt.xlabel("Layer index (Conv2D layer)")
    plt.ylabel("Batch normalization gamma creterion")
    plt.show()

    return gamma_p, model_params


def get_and_plot_channel_distribution(original_model, modified_model):
    original_channels, original_inputs = get_channel_numbers(original_model)
    new_channels, new_inputs = get_channel_numbers(modified_model)

    x = list(original_channels.keys())
    x = np.asarray([float(i) for i in x])

    y1 = list(original_channels.values())
    y2 = list(original_inputs.values())
    y3 = list(new_channels.values())
    y4 = list(new_inputs.values())

    y5 = []
    y6 = []
    for n in range(len(y1)):
        y5.append((float(y3[n]))/(float(y1[n])))
        y6.append((float(y4[n]))/(float(y2[n])))

    plt.bar(x+0.0, y6, color='b', width=0.15)
    plt.bar(x+0.15, y5, color='r', width=0.15)
    plt.show()


def plot_sensitive_graph(model,  prune_ratio=0.5):
    filter_sum = []
    filters_all = []
    filter_values = []
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
            filters = []
            filter = layer.get_weights()
            for i in range(filter[0].shape[-1]):
                filters.append(np.sum(np.abs(filter[0][:, :, :, i])))

            filter_sum.append(filters)
            filter_values.append(np.asarray(filter[0]).flatten())
            filters_all = np.concatenate([filters_all, np.asarray(filter[0]).flatten()])
    filters_sum_all = np.sort(np.abs(filters_all))
    prune_threshold = filters_sum_all[int(prune_ratio*len(filters_sum_all))]

    # how many percentage are left in each layer
    filter_all_mask = []
    for filters in filter_values:
        filter_group_mask = np.abs(filters) > prune_threshold
        filter_all_mask.append(np.sum(filter_group_mask)/len(filter_group_mask))
    # figs, axs = plt.subplots(2)
    # axs[0].plot(filter_all_mask)

    for index, filter in enumerate(filter_sum):
        filter = np.sort(filter)
        filter = filter[::-1]
        filter = filter/np.max(np.abs(filter), axis=0)

        x = np.linspace(0, 100, num=len(filter))
        if index < 7:
            plt.plot(x, filter, label=f"{index} layer")
        else:
            plt.plot(x, filter, '--', label=f"{index} layer")

    plt.legend()
    plt.show()


def get_pruned_ratio(raw_model, pruned_model):
    pruned_ratio = {}
    for index, layer in enumerate(raw_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            original_channels = layer.weights[0].shape[-1]
            pruned_channels = pruned_model.layers[index].weights[0].shape[-1]
            pruned_ratio[index] = (original_channels-pruned_channels)/original_channels
    return pruned_ratio
