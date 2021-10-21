import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorflow.keras import models
import torch


def VH_decomposition_conv_layer(
        layers,
        rank=None,
        ):
    tl.set_backend("tensorflow")
    # First time decomposition
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    rank = rank[0]
    layer_data = tf.transpose(layer_data, [3, 0, 2, 1])
    layer_data = np.asarray(layer_data)
    dim = layer_data.shape
    layer_data = layer_data.reshape(dim[0]*dim[1], -1)
    layer_data = torch.tensor(layer_data)
    V, sigmaVH, H = torch.svd(layer_data)
    V = V[:, :rank]
    H = np.transpose(H)
    H = H[:rank, :]
    sigmaVH = sigmaVH[:rank]
    H = np.diag(sigmaVH).dot(H)
    H = H.reshape([rank, dim[2], dim[3], 1])
    H = np.transpose(H, [3, 2, 1, 0])
    V = V.reshape([dim[0], 1, dim[1], rank])
    V = np.transpose(V, [2, 1, 3, 0])
    print(f"V has shape {V.shape}, H has shape {H.shape}")
    new_layers = from_tensor_to_layers([V, H], layer, bias)
    return new_layers


def from_tensor_to_layers(
        tensors,
        layers,
        bias,
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

    layer = layers
    [V, H] = tensors
    bias = layer.get_weights()[1] if layer.use_bias else None
    first_layer = tf.keras.layers.Conv2D(
                    name=layer.name+"V",
                    filters=H.shape[3], kernel_size=[H.shape[0], H.shape[1]],
                    strides=layer.strides, padding=(layer.padding),
                    dilation_rate=layer.dilation_rate, use_bias=False,
                    input_shape=layer.input_shape[1:])
    last_layer = tf.keras.layers.Conv2D(
                    name=layer.name+"H",
                    filters=V.shape[3], kernel_size=[V.shape[0], V.shape[1]],
                    padding=(layer.padding), dilation_rate=layer.dilation_rate,
                    use_bias=layer.use_bias,  activation=layer.activation)
    l_model = models.Sequential()
    l_model.add(first_layer)
    l_model.add(last_layer)
    l_model.build()

    first_layer.set_weights([H])
    if layer.use_bias:
        last_layer.set_weights([V, bias])
    else:
        last_layer.set_weights([V])

    new_layer = [first_layer, last_layer]

    return new_layer
