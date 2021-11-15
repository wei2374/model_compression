import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorflow.keras import models
import torch


def channel_decomposition_all_conv_layer(
        layers,
        ranks=None,
        ):
    tl.set_backend("tensorflow")

    # First time decomposition
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    # print(f"Original input channels is : {layer_data.shape[1]},\
    #     Estimate input channels is : {ranks[1]}")
    # print(f"Original output channels is : {layer_data.shape[0]},\
    #     Estimate output channels is : {ranks[0]}")

    rank = ranks[0]
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    layer_data = layer_data.reshape(dim[0]*dim[1]*dim[2], -1)
    layer_data = torch.tensor(layer_data)
    N, sigmaVH, C = torch.svd(layer_data)
    rank = rank if rank < N.shape[1] else N.shape[1]
    N = N[:, :rank]
    C = np.transpose(C)
    C = C[:rank, :]
    sigmaVH = sigmaVH[:rank]
    sigmaVH = torch.sqrt(sigmaVH)
    N = np.asarray(N).dot(np.diag(sigmaVH))
    C1 = np.diag(sigmaVH).dot(C)
    N = N.reshape(dim[0], dim[1], dim[2], rank)
    C1 = np.transpose(C1)

    rank = ranks[1]
    new_layer_data = N
    new_layer_data = np.transpose(new_layer_data, [0, 1, 3, 2])
    dim = new_layer_data.shape
    new_layer_data = new_layer_data.reshape(dim[0]*dim[1]*dim[2], -1)
    layer_data = torch.tensor(new_layer_data)
    N, sigmaVH, C = torch.svd(layer_data)
    rank = rank if rank < N.shape[1] else N.shape[1]
    N = N[:, :rank]
    C = np.transpose(C)
    C = C[:rank, :]
    sigmaVH = sigmaVH[:rank]
    sigmaVH = torch.sqrt(sigmaVH)
    N = np.asarray(N).dot(np.diag(sigmaVH))
    C2 = np.diag(sigmaVH).dot(C)
    N = N.reshape(dim[0], dim[1], dim[2], rank)
    N = np.transpose(N, [2, 3, 0, 1])
    C2 = np.transpose(C2)

    new_layers = from_tensor_to_layers([C2, C1, N], layer, bias)
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

    Return:
    new_layers
    '''
    [first, last, core] = tensors
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

    F = tf.expand_dims(tf.expand_dims(
        first, axis=0, name=None
        ), axis=0, name=None)
    C = np.transpose(core, [2, 3, 1, 0])

    L = tf.expand_dims(tf.expand_dims(
        np.transpose(last), axis=0, name=None
        ), axis=0, name=None)
    new_weights = [F, C, L]
    
    if layer.use_bias:
        new_weights.append(bias)
        
    new_layer = [first_layer, core_layer, last_layer]
    return new_layer, new_weights

