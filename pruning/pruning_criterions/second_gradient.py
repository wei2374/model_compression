import tensorflow as tf
import numpy as np
from dataset.load_datasets import get_data_from_dataset
from tools.progress.bar import Bar


def get_and_plot_gradients2(model, layer_index_dic, dataset="food101", PLOT=False):
    train_data, _ = get_data_from_dataset(dataset, batch_size=4)
    batches_n = 10
    conv2d_layers = {}
    c = 0

    # The target is the output of all Conv2D layers
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) and \
           not isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            # next_layer = layer.outbound_nodes[0].layer
            # next_layer = next_layer.outbound_nodes[0].layer
            # if isinstance(next_layer, tf.keras.layers.ReLU):
            conv2d_layers[index] = layer
            c += 1

    crits = {}
    with Bar(f'Channel importance estimation based on second-order gradient...') as bar:
        for b in range(batches_n):
            batch = next(train_data)
            grad = {}
            # Get gradients
            for conv_index in conv2d_layers:
                Input = tf.convert_to_tensor(batch[0])
                with tf.GradientTape(persistent=True) as t:
                    t.watch(Input)
                    with tf.GradientTape(persistent=True) as tt:
                        ##########################################
                        # run the following layers
                        tt.watch(Input)
                        Tensor_data = Input
                        pred = model(Tensor_data)
                        ###########################################
                        loss = tf.metrics.categorical_crossentropy(batch[1], pred)
                    grad1 = tt.gradient(loss, model.layers[conv_index].weights[0])
                grad[conv_index] = t.gradient(grad1, model.layers[conv_index].weights[0])

            # Compute criterion
            for n in grad:
                crit = np.multiply(grad[n], conv2d_layers[n].weights[0])
                # crit = np.asarray(grad[n])
                channels = crit.shape[3]
                inputs = crit.shape[0]
                crit = crit.transpose(3, 0, 1, 2)
                crit = np.abs(crit.reshape(channels, inputs, -1))
                cirt_av = np.average(np.abs(np.average(np.abs(crit), axis=2)), axis=1)
                if b == 0:
                    crits[n] = cirt_av/batches_n
                else:
                    crits[n] += cirt_av/batches_n
            bar.next((100/(batches_n)))

    creterions = {}
    for layer in crits:
        creterions[layer] = np.linalg.norm(crits[layer])
        crits[layer] = crits[layer]/creterions[layer]

    critss = crits
    return critss
