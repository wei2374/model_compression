import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorflow.keras import models
import scipy
from pruning.pruning_methods.lasso_pruning import extract_inputs_and_outputs
from pruning.helper_functions import load_model_param


def channel_decomposition_nl_conv_layer(
        original_model,
        index,
        layers,
        rank=None,
        ):
    tl.set_backend("tensorflow")

    # First time decomposition
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    # print(f"Original output channels is : {layer_data.shape[-1]},\
    #     Estimated output channels is : {rank[0]}")

    rank = rank[0]
    W1, W2, B = get_actapp_layer_data(index, original_model, layer_data, rank)
    bias = B,
    print(f"N has shape {W1.shape}, C has shape {W2.shape}")
    new_layers = from_tensor_to_layers([W2, W1], layer, bias, method="channel")
    return new_layers


# TODO::Zhang's method (2015), not work yet
def rel_error(A, B):
    return np.mean((A - B)**2)**.5 / np.mean(A**2)**.5


def svd(x):
    return scipy.linalg.svd(x, full_matrices=False, lapack_driver='gesvd')


def relu(x):
    return np.maximum(x, 0.)


def pinv(x):
    import scipy
    return scipy.linalg.pinv(x, 1e-6)


def get_actapp_layer_data(index, original_model, layer_data, rank):
    layer = original_model.layers[index]
    dataset = "food20"
    _, _, _, _, layer_index_dic = load_model_param(original_model)
    [inputs, outputs] = extract_inputs_and_outputs(
                            original_model,
                            layer,
                            layer_index_dic,
                            dataset=dataset)
    X = inputs
    Y = outputs
    Y_mean = np.average(np.asarray(Y), axis=0)
    Z = relu(Y)
    G = Y - Y_mean
    G = G.T
    X = G.dot(G.T)
    L, sigma, R = svd(X)
    L = np.asarray(L)
    sigma = np.asarray(sigma)
    R = np.asarray(R)
    T = L[:, :rank].dot(np.diag(sigma[:rank])).dot(R[:rank, :])
    L, sigma, R = svd(T)
    # L, sigma, R = np.linalg.svd(T,0)
    L = L[:, :rank]
    # R = np.transpose(R)
    R = R[:rank, :]
    R = np.diag(sigma[:rank]).dot(R)

    weight = layer_data
    dim = weight.shape
    W1 = np.asarray(weight).reshape([-1, dim[3]]).dot(L)
    W1 = W1.reshape([dim[0], dim[1], dim[2], rank])
    W2 = R
    # W2 = W2.T
    W2 = tf.expand_dims(tf.expand_dims(
            W2, axis=0, name=None
            ), axis=0, name=None)
    # W2 = W2.reshape(1, 1, rank, dim[3])
    # B = - Y_mean.dot(T) + Y_mean
    if layer.use_bias:
        B = layer.bias
    return W1, W2, B


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
