import tensorflow as tf
import numpy as np
import torch


def energy_threshold(sigma, param):
    energy = sigma
    sum_e = torch.sum(energy)
    for i in range(energy.size(0)):
        if energy[:(i+1)].sum()/sum_e >= param:
            valid_idx = i+1
            break

    return valid_idx


def estimate_ranks_energy(layer, param=0.9):
    layer_data = layer.weights[0]
    layer_data = tf.transpose(layer_data, [3, 0, 1, 2])
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    W = layer_data.reshape(dim[0], -1)
    W = torch.tensor(W)
    U, sigma, V = torch.svd(W)
    c_out = energy_threshold(sigma, param)

    layer_data = layer.weights[0]
    layer_data = tf.transpose(layer_data, [2, 0, 1, 3])
    dim = layer_data.shape
    layer_data = np.asarray(layer_data)
    W = layer_data.reshape(dim[0], -1)
    W = torch.tensor(W)
    U, sigma, V = torch.svd(W)
    c_in = energy_threshold(sigma, param)
    return c_out, c_in
