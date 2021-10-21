import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorly.decomposition import parafac
from tensorflow.keras import models


def cp_decomposition_conv_layer(
        layers,
        rank=None
        ):
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    rank = rank[0]
    tl.set_backend("pytorch")
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    vertical, horizontal, first, last = \
        parafac(layer_data, rank=rank, init='random')[1]

    new_layers = from_tensor_to_layers(
            [vertical, horizontal, first, last],
            layer,
            bias)

    return new_layers


def from_tensor_to_layers(
        tensors,
        layers,
        bias
        ):
    '''
    transform tensors to layers

    Key arguments:
    tensors -- contains data of decomposed layer
    layers -- original layers
    bias -- bias of layer

    Return:
    new_layers
    '''
    layer = layers
    [vertical, horizontal, first, last] = tensors
    pointwise_s_to_r_layer = tf.keras.layers.Conv2D(
                        name=layer.name+"p1",
                        filters=first.shape[1],
                        kernel_size=[1, 1],
                        padding="valid",
                        use_bias=False, input_shape=layer.input_shape[1:])

    depthwise_vertical_layer = tf.keras.layers.DepthwiseConv2D(
                        name=layer.name+"v",
                        kernel_size=[vertical.shape[0], 1], strides=layer.strides,
                        padding=layer.padding, dilation_rate=layer.dilation_rate,
                        use_bias=False)

    depthwise_horizontal_layer = tf.keras.layers.DepthwiseConv2D(
                        name=layer.name+"h",
                        kernel_size=[1, horizontal.shape[0]], strides=layer.strides,
                        padding=layer.padding, dilation_rate=layer.dilation_rate,
                        use_bias=False)

    pointwise_r_to_t_layer = tf.keras.layers.Conv2D(
                        name=layer.name+"p2",
                        filters=last.shape[0],
                        kernel_size=[1, 1], use_bias=layer.use_bias,
                        padding="valid",
                        activation=layer.activation)

    l_model = models.Sequential()
    l_model.add(pointwise_s_to_r_layer)
    l_model.add(depthwise_vertical_layer)
    l_model.add(depthwise_horizontal_layer)
    l_model.add(pointwise_r_to_t_layer)
    l_model.build()

    # This section assign weights to the layers
    H = tf.expand_dims(tf.expand_dims(
        horizontal, axis=0, name=None
    ), axis=2, name=None)
    H = np.transpose(H, (0, 1, 3, 2))
    depthwise_horizontal_layer.set_weights([H])

    V = tf.expand_dims(tf.expand_dims(
        vertical, axis=1, name=None
    ), axis=1, name=None)
    V = np.transpose(V, (0, 1, 3, 2))
    depthwise_vertical_layer.set_weights([V])

    F = tf.expand_dims(tf.expand_dims(
        first, axis=0, name=None
    ), axis=0, name=None)
    pointwise_s_to_r_layer.set_weights([F])

    L = tf.expand_dims(tf.expand_dims(
        np.transpose(last), axis=0, name=None
    ), axis=0, name=None)
    if layer.use_bias:
        pointwise_r_to_t_layer.set_weights([L, bias])
    else:
        pointwise_r_to_t_layer.set_weights([L])
    new_layer = [
        pointwise_s_to_r_layer, depthwise_vertical_layer,
        depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return new_layer
