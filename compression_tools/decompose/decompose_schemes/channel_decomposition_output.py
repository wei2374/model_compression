import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorflow.keras import models
import torch


def output_channel_decomposition_conv_layer(
        layers,
        rank=None,
        ):
    tl.set_backend("tensorflow")

    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)

    # print(f"Original output channels is : {layer_data.shape[-1]},\
    #     Estimate output channels is : {rank[0]}")

    rank = rank[0]
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
    C = np.diag(sigmaVH).dot(C)
    N = N.reshape(dim[0], dim[1], dim[2], rank)
    C = C.reshape(1, 1, rank, dim[3])
    print(f"N has shape {N.shape}, C has shape {C.shape}")
    new_layers = from_tensor_to_layers([C, N], layer, bias)
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

    layer = layers
    [V, H] = tensors
    bias = layer.get_weights()[1] if layer.use_bias else None
    first_layer = tf.keras.layers.Conv2D(
                    name=layer.name+"first",
                    filters=H.shape[3], kernel_size=[H.shape[0], H.shape[1]],
                    strides=layer.strides, padding=(layer.padding),
                    dilation_rate=layer.dilation_rate, use_bias=False,
                    input_shape=layer.input_shape[1:])
    last_layer = tf.keras.layers.Conv2D(
                    name=layer.name+"last",
                    filters=V.shape[3], kernel_size=[V.shape[0], V.shape[1]],
                    padding=(layer.padding), dilation_rate=layer.dilation_rate,
                    use_bias=layer.use_bias,  activation=layer.activation)

    new_weight = [H, V]
    if layer.use_bias:
        new_weight.append(bias)

    new_layer = [first_layer, last_layer]
    return new_layer, new_weight
