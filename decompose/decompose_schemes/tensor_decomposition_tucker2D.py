import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorflow.keras import models
from tensorly.decomposition import partial_tucker


def tucker_decomposition_conv_layer(
        layers,
        rank=None,
        ):
    '''
    This function performs tucker2D decomposition to a layer
    Key arguments:
    layers -- the list of layers to be composed
              the size of layers is 1 if decompose_time is 1
              the size of layers is 3 if decompose_time is larger than 1
    rank -- the ranks to be decomposed to
    decomposed_time -- if the model is decomposed for the first time
    Return:
    new_layers -- new layers
    '''
    tl.set_backend("tensorflow")
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

    # print(f"Original input channels is : {layer_data.shape[1]},\
    #     Estimate input channels is : {rank[1]}")
    # print(f"Original output channels is : {layer_data.shape[0]},\
    #     Estimate output channels is : {rank[0]}")

    core, [last, first] = partial_tucker(
                                layer_data,
                                modes=[0, 1],
                                rank=rank,
                                init='svd')

    core = tl.tensor(core)
    first = tl.tensor(first)
    last = tl.tensor(last)
    new_layers = from_tensor_to_layers(
                            [first, last, core],
                            layer,
                            bias)

    # TODO::Gradually decompose decompoed layers
    '''
    else:
    raise Exception("not implemented yet")
    core_layer = layers[1]
    input_layer = layers[0]
    output_layer = layers[2]

    core_weights = np.asarray(core_layer.get_weights()[0])
    input_weights = np.asarray(input_layer.get_weights()[0])
    output_weights = np.asarray(output_layer.get_weights()[0])
    bias = output_layer.get_weights()[1]

    core_layer_data = tl.tensor(core_weights)
    core_layer_data = tf.transpose(core_layer_data, [3, 2, 0, 1])
    input_layer_data = tl.tensor(input_weights)
    input_layer_data = tf.transpose(input_layer_data, [3, 2, 0, 1])
    output_layer_data = tl.tensor(output_weights)
    output_layer_data = tf.transpose(output_layer_data, [3, 2, 0, 1])

    print(f"Original ranks are : {core_layer_data.shape}")

    # Rank estimation
    if rank is None:
        if rank_selection == "VBMF":
            rank = estimate_ranks_VBMF(core_layer)
        elif rank_selection == "weak_VNMF":
            rank = estimate_ranks_VBMF(core_layer, vbmf_weakenen_factor=0.7)
        elif rank_selection == "BayesOpt":
            rank = estimate_ranks_BayesOpt(core_layer)
        elif rank_selection == "Param":
            rank = estimate_ranks_param(core_layer)

        print(f"Estimated ranks are : {rank}")

    core, [last, first] = partial_tucker(core_layer_data, \
        modes=[0, 1], rank=rank, init='svd')
    input_layer_data = tf.linalg.matmul(np.transpose(tf.squeeze(input_layer_data,\
            axis=[2,3])), first)
    output_layer_data = tf.linalg.matmul( (tf.squeeze(output_layer_data, axis=[2,3]))\
        ,(last))
    new_layers = from_tensor_to_layers([tl.tensor(input_layer_data),\
        tl.tensor(output_layer_data), tl.tensor(core)], layers, bias, decompose_time)
    '''
    return new_layers


def from_tensor_to_layers(
        tensors,
        layers,
        bias,
        decompose_time=1
        ):
    '''
    transform tensors to layers

    Key arguments:
    tensors -- contains data of decomposed layer
    layers -- original layers
    bias -- bias of layer
    decomposed_time -- if the model is decomposed for the first time

    Return:
    new_layers
    '''
    [first, last, core] = tensors
    if decompose_time == 1:
        layer = layers
        first_layer = tf.keras.layers.Conv2D(
                        name=layer.name+"first",
                        filters=first.shape[1], kernel_size=1,
                        strides=1, dilation_rate=layer.dilation_rate, use_bias=False,
                        input_shape=layer.input_shape[1:])

        core_layer = tf.keras.layers.Conv2D(
                        name=layer.name+"core",
                        filters=core.shape[0], kernel_size=layer.kernel_size,
                        strides=layer.strides, padding=layer.padding,
                        dilation_rate=layer.dilation_rate, use_bias=False)
        last_layer = tf.keras.layers.Conv2D(
                        name=layer.name+"last",
                        filters=last.shape[0], kernel_size=1,
                        strides=1, dilation_rate=layer.dilation_rate,
                        use_bias=layer.use_bias, activation=layer.activation)

    # TODO::Gradually decompose decompoed layers
    # else:
    #     layer = layers[1]
    #     first_layer = tf.keras.layers.Conv2D(
    #                     filters=first.shape[1], kernel_size=1,
    #                     strides=1, dilation_rate=layer.dilation_rate, use_bias=False,
    #                     input_shape=layers[0].input_shape[1:])
    #     core_layer = tf.keras.layers.Conv2D(
    #                 filters=core.shape[0], kernel_size=layer.kernel_size,
    #                 strides=layer.strides, padding=layer.padding,
    #                 dilation_rate=layer.dilation_rate, use_bias=False)
    #     last_layer = tf.keras.layers.Conv2D(
    #                 filters=last.shape[0], kernel_size=1, strides=1,
    #                 dilation_rate=layer.dilation_rate, use_bias=layer.use_bias,
    #                 activation=layer.activation)

    l_model = models.Sequential()
    l_model.add(first_layer)
    l_model.add(core_layer)
    l_model.add(last_layer)
    l_model.build()

    F = tf.expand_dims(tf.expand_dims(
        first, axis=0, name=None
        ), axis=0, name=None)
    first_layer.set_weights([F])

    L = tf.expand_dims(tf.expand_dims(
        np.transpose(last), axis=0, name=None
        ), axis=0, name=None)

    if layer.use_bias:
        last_layer.set_weights([L, bias])
    else:
        last_layer.set_weights([L])

    core_layer.set_weights([np.transpose(core, [2, 3, 1, 0])])
    new_layer = [first_layer, core_layer, last_layer]

    return new_layer
