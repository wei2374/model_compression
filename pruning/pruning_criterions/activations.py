import tensorflow as tf
import numpy as np
from pruning.helper_functions import map_act_to_conv
from dataset.load_datasets import get_data_from_dataset
from tools.progress.bar import Bar


def get_and_plot_activations(model, layer_index_dic, dataset="food20"):
    train_data, _ = get_data_from_dataset(dataset)
    batches_n = 40
    crits_act = {}
    activation_layers = {}
    c = 0

    # The target is the output of all activation layers
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Activation) \
           or isinstance(layer, tf.keras.layers.ReLU):
            activation_layers[index] = layer
            c += 1

    with Bar(f'Channel importance estimation based on magnitude of activation output...') as bar:
        for b in range(batches_n):
            batch = next(train_data)
            # Get activation layer output
            intermediate_layer_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[activation_layers[layer].output for layer in activation_layers])

            # TODO::to be optimized
            if c == 1:
                activation_outputs = []
                activation_outputs.append(intermediate_layer_model(batch[0]))
            else:
                activation_outputs = (intermediate_layer_model(batch[0]))

            activation_outputs_dic = {}
            for counter, layer_index in enumerate(activation_layers):
                activation_outputs_dic[layer_index] = activation_outputs[counter]
                activation_out = np.asarray(activation_outputs_dic[layer_index])
                in_channels = activation_out.shape[0]
                out_channels = activation_out.shape[-1]
                activation_out = activation_out.transpose(3, 0, 1, 2).reshape(
                                out_channels, in_channels, -1)
                activation_out_std_abs_av = np.average(
                    np.abs(np.average(activation_out, axis=2)), axis=1)

                if b == 0:
                    crits_act[layer_index] = activation_out_std_abs_av/batches_n
                else:
                    crits_act[layer_index] += activation_out_std_abs_av/batches_n
            bar.next(int(100/(batches_n)))

    for layer_index in crits_act:
        crits_normalize = np.linalg.norm(crits_act[layer_index])
        crits_act[layer_index] = crits_act[layer_index]/crits_normalize

    # TODO::map act to conv
    crits = map_act_to_conv(activation_layers, crits_act, layer_index_dic)
    return crits
