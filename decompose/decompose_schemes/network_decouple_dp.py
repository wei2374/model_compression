import numpy as np
import tensorly as tl
import tensorflow as tf
from decompose.rank_estimation.rank_estimate_energy import energy_threshold
import torch


def network_decouple_conv_layer_dp(
        layers,
        param,
        rank=None,
        ):
    tl.set_backend("tensorflow")
    # First time decomposition
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [2, 3, 0, 1])
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    layer_data = torch.tensor(layer_data)
    valid_idx = []
    for i in range(dim[0]):
        W = layer_data[i, :, :, :].reshape(dim[1], -1)
        W = torch.tensor(W)
        U, sigma, V = torch.svd(W)
        valid_idx.append(energy_threshold(sigma, param))
    # if rank is None:
    item_num = min(max(valid_idx), min(dim[2]*dim[3], dim[1]))
    # else:
    #   item_num = rank[0]
    pw = [np.zeros((dim[0], dim[1], 1, 1)) for i in range(item_num)]
    dw = [np.zeros((dim[0], 1, dim[2], dim[3])) for i in range(item_num)]
    # print(f"Breaks into {item_num} conv channels")

    # svd decoupling
    for i in range(dim[0]):
        W = layer_data[i, :, :, :].view(dim[1], -1)
        W = torch.tensor(W)
        U, sigma, V = torch.svd(W, some=True)
        V = V.t()
        U = U[:, :item_num].contiguous()
        V = V[:item_num, :].contiguous()
        sigma = torch.diag(torch.sqrt(sigma[:item_num]))
        U = U.mm(sigma)
        V = sigma.mm(V)
        V = V.view(item_num, dim[2], dim[3])
        for j in range(item_num):
            pw[j][i, :, 0, 0] = U[:, j]
            dw[j][i, 0, :, :] = V[j, :, :]

    new_layers = from_tensor_to_layers([pw, dw], layer, bias)
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
    [pw,  dw] = tensors
    input, output, _, _ = pw[0].shape
    _, _, kernel_size, _ = dw[0].shape
    item_number = len(pw)
    p_layers = []
    d_layers = []
    for i in range(item_number):
        p_layers.append(tf.keras.layers.Conv2D(
                    name=layer.name+f"p{i}",
                    filters=output, kernel_size=[1, 1], strides=(1, 1),
                    use_bias=(i == 0 and layer.use_bias)))
        d_layers.append(tf.keras.layers.DepthwiseConv2D(
                    name=layer.name+f"d{i}",
                    kernel_size=[kernel_size, kernel_size],
                    strides=(layer.strides), padding=(layer.padding),
                    dilation_rate=layer.dilation_rate,
                    activation=layer.activation, use_bias=None
                    ))

    new_weight_p = []
    new_weight_d = []
    for i in range(item_number):
        new_weight_d.append(np.transpose(dw[i], [2, 3, 0, 1]))

        if i == 0:
            if layer.use_bias:
                new_weight_p.append(np.transpose(pw[i], [2, 3, 0, 1]))
                new_weight_p.append(np.transpose(bias))
            else:
                new_weight_p.append(np.transpose(pw[i], [2, 3, 0, 1]))
        else:
            new_weight_p.append(np.transpose(pw[i], [2, 3, 0, 1]))

    return [p_layers, d_layers], [new_weight_p, new_weight_d]
