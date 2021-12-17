import numpy as np
import tensorly as tl
import tensorflow as tf
from sparsity_estimation.rank_estimator import energy_threshold
from tensorflow.keras import models
import torch


def decompose_dense_layer(
        layers,
        param,
        rank=None,
        ):
    tl.set_backend("tensorflow")
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1]
    layer_data = tl.tensor(weights)
    print(f"Original ranks are : {layer_data.shape}")

    layer_data = np.asarray(layer_data)
    layer_data = torch.tensor(layer_data)
    N, sigma, C = torch.svd(layer_data)
    rank = energy_threshold(sigma, param)
    N = N[:, :rank]
    C = np.transpose(C)
    C = C[:rank, :]
    sigma = sigma[:rank]
    sigma = torch.sqrt(sigma)
    N = np.asarray(N).dot(np.diag(sigma))
    C = np.diag(sigma).dot(C)

    new_layers = from_tensor_to_layers(
                [N, C],
                layer,
                bias,
                method="Dense",
                decompose_time=1)

    return new_layers


def from_tensor_to_layers(
        tensors,
        layers,
        bias,
        method,
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

    layer = layers
    [V, H] = tensors
    bias = layer.get_weights()[1]

    first_layer = tf.keras.layers.Dense(
                units=V.shape[1], input_shape=layer.input_shape,
                use_bias=False)
    last_layer = tf.keras.layers.Dense(
                units=H.shape[1], use_bias=True,
                activation=layer.activation)

    l_model = models.Sequential()
    l_model.add(first_layer)
    l_model.add(last_layer)
    l_model.build()
    first_layer.set_weights([V])
    last_layer.set_weights([H, bias])
    new_layer = [first_layer, last_layer]
    return new_layer
