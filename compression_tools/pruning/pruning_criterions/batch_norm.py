import tensorflow as tf
import numpy as np
from compression_tools.pruning.helper_functions import get_layer_index
from compression_tools.pruning.helper_functions import map_act_to_conv


def get_and_plot_BN(model, layer_index_dic, PLOT=False):
    crits = {}
    bn_layers = {}
    # Get out the gamma of all BN layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer_index = get_layer_index(layer_index_dic, layer)
            layer_weights = np.asarray(layer.get_weights())
            crits[layer_index] = layer_weights[0]
            bn_layers[layer_index] = layer

    crits = map_act_to_conv(bn_layers, crits, layer_index_dic)
    # DEBUG::how many percentage are left in each layer
    return crits
