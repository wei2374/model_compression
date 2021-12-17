import tensorflow as tf
import numpy as np
from compression_tools.pruning.helper_functions import get_block_result
from compression_tools.pruning.helper_functions import map_act_to_conv
from tools.progress.bar import Bar


def get_and_plot_taylor1(model, layer_index_dic, get_dataset=None, PLOT=False):
    train_data, _ = get_dataset()
    batches_n = 10
    activation_layers = {}
    c = 0

    # The target is the output of all activation layers
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Activation) or isinstance(layer, tf.keras.layers.ReLU):
            activation_layers[index] = layer
            c += 1

    crits = {}
    with Bar(f'Channel importance estimation based on first-order taylor expansion...') as bar:
        for b in range(batches_n):
            if isinstance(train_data, tf.data.Dataset):
                batch = next(iter(train_data))
            else:
                batch = next(train_data)
                
            intermediate_layer_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=[activation_layers[layer].output for layer in activation_layers])
            if c == 1:
                activation_outputs = []
                activation_outputs.append(intermediate_layer_model(batch[0]))
            else:
                activation_outputs = []
                activation_outputs = intermediate_layer_model(batch[0])

            activation_outputs_dic = {}
            for counter, layer_index in enumerate(activation_layers):
                activation_outputs_dic[layer_index] = activation_outputs[counter]
            activation_outputs = []

            grad = {}
            # Get gradients
            if batch[1].ndim==1:
                loss_fun = "tf.metrics.sparse_categorical_crossentropy(batch[1], pred)"
            else:
                loss_fun = "tf.metrics.categorical_crossentropy(batch[1], pred)"
                
            for act_index in activation_outputs_dic:
                Input = tf.convert_to_tensor(activation_outputs_dic[act_index])
                with tf.GradientTape(persistent=True) as t:
                    t.watch(Input)
                    ##########################################
                    # run the following layers
                    Tensor_data = Input
                    act_layer = activation_layers[act_index]

                    continue_flag = True
                    while(continue_flag):
                        act_layer, Tensor_data, continue_flag = get_block_result(
                                                    act_layer, Tensor_data)
                    pred = Tensor_data
                    ###########################################
                    loss = eval(loss_fun)
                grad[act_index] = t.gradient(loss, Input)

            # Compute criterion
            for n in grad:
                crit = np.multiply(grad[n], activation_outputs_dic[n])
                channels = crit.shape[3]
                inputs = crit.shape[0]
                crit = crit.transpose(3, 0, 1, 2)
                crit = crit.reshape(channels, inputs, -1)
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

    critss = map_act_to_conv(activation_layers, crits, layer_index_dic)    

    return critss