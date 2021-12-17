import numpy as np
import tensorly as tl
import tensorflow as tf
import torch


def mono_conv_approximation(layers):
    U_buffer = []
    V_buffer = []
    layer = layers[0]
    weights = np.asarray(layer.get_weights()[0])
    bias = layer.get_weights()[1] if layer.use_bias else None
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
    c_out, c_in, height, width = layer_data.shape
    for filter_weight in layer_data:
        filter_weight = np.reshape(filter_weight, (c_in, -1))
        filter_weight = torch.tensor(filter_weight)
        N, sigmaVH, C = torch.svd(filter_weight)
        rank = N.shape[1]
        N = N[:, :rank]
        C = np.transpose(C)
        C = C[:rank, :]
        sigmaVH = sigmaVH[:rank]
        C = np.diag(sigmaVH).dot(C)
        U_buffer.append(np.asarray(np.reshape(N, -1)))
        V_buffer.append(C)
    from sklearn.cluster import KMeans
    kmeans = KMeans(
        init="random",
        n_clusters=int(c_out/2),
        max_iter=3000,
    )
    kmeans.fit(U_buffer)
    # print(kmeans.cluster_centers_)
    center = kmeans.cluster_centers_
    new_filter_weights = []
    for c in range(c_out):
        # new_filter_weight = (np.reshape(U_buffer[c], (c_in, 1))).dot(V_buffer[c])
        label = kmeans.labels_[c]
        new_filter_weight = np.reshape(center[label], (N.shape[0], N.shape[1])).dot(V_buffer[c])
        new_filter_weights.append(new_filter_weight)
    new_filter_weights = np.reshape(new_filter_weights, (c_out, c_in, height, width))
    layer_data = tf.transpose(new_filter_weights, [2, 3, 1, 0])
    layers[0].set_weights([np.asarray(layer_data), bias])
    return layers
