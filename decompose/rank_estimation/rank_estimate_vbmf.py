import tensorly as tl
import tensorflow as tf
import numpy as np
from . import VBMF


def estimate_ranks_VBMF(
            layer,
            schema="tucker2D",
            noise_variance=None
        ):
    '''
    VBMF rank estimation

    Key arguments:
    layer -- layer to be decomposed
    min_rank -- if estimated_rank is below min_rank, the min_rank will be used instead
    schema -- decompose schema
    noise_variance -- noise varaince of original matrix, use auto VBMF estimation if
                      is set to None

    Return:
    ranks_weaken -- ranks estimated
    '''

    tl.set_backend('tensorflow')

    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)

    if schema == "tucker2D" or schema == "whole_channel":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_0 = tl.base.unfold(layer_data, 0)
        unfold_1 = tl.base.unfold(layer_data, 1)

        if noise_variance is None:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        else:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, noise_variance)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1, noise_variance)

        ranks = [diag_0.shape[0], diag_1.shape[1]]

        return ranks

    if schema == "VH":
        layer_data = tf.transpose(layer_data, [2, 0, 3, 1])
        layer_data = np.asarray(layer_data)
        layer_shape = layer_data.shape
        unfold_0 = layer_data.reshape(layer_shape[0]*layer_shape[1], -1)
        if noise_variance is None:
            # print("Auto VBMF rank estimation")
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        else:
            # print(f"Estimate rank based on noise variance {noise_variance}")
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, noise_variance)

        ranks = [diag_0.shape[0]]
        return ranks

    if schema == "output_channel" or schema == "nl_channel":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_0 = tl.base.unfold(layer_data, 0)

        if noise_variance is None:
            # print("Auto VBMF rank estimation")
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        else:
            # print(f"Estimate rank based on noise variance {noise_variance}")
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, noise_variance)

        ranks = [diag_0.shape[0]]
        return ranks

    if schema == "input_channel":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_1 = tl.base.unfold(layer_data, 1)

        if noise_variance is None:
            # print("Auto VBMF rank estimation")
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        else:
            # print(f"Estimate rank based on noise variance {noise_variance}")
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1, noise_variance)

        ranks = [diag_1.shape[1]]

        return ranks
