import tensorflow as tf
import numpy as np
from tools.progress.bar import Bar


def get_and_plot_weights(model):
    crits = {}
    # Get out the weights of all Conv2D layers
    target_layers = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
            target_layers[index] = layer

    with Bar(f'Channel importance estimation based on magnitude of weights...') as bar:
        for index in target_layers:
            layer_weights = np.asarray(target_layers[index].get_weights()[0])
            channels = layer_weights.shape[3]
            channel_weights = layer_weights.transpose(3, 0, 1, 2).reshape(channels, -1)
            channel_weights_abs_av = np.average(np.abs(channel_weights), axis=1)
            crits[index] = channel_weights_abs_av
            bar.next((100/len(target_layers)))

    for layer_index in crits:
        crits_norm = np.linalg.norm(crits[layer_index])
        crits[layer_index] = crits[layer_index]/crits_norm

    return crits
