import tensorly as tl
import tensorflow as tf
import numpy as np
from . import VBMF


def estimate_ranks_VBMF(
            layer,
            schema="tucker2D",
            factor=None
        ):
    '''
    VBMF rank estimation : Select the rank such as the noise variance estimation is the same for each layer

    Key arguments:
    layer -- layer to be decomposed
    schema -- decompose schema
    factor -- noise varaince of original matrix, use auto VBMF estimation if
                      is set to None

    Return:
    ranks estimated
    '''

    tl.set_backend('tensorflow')

    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)

    if schema is not "VH":
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])
        unfold_0 = tl.base.unfold(layer_data, 0)
        unfold_1 = tl.base.unfold(layer_data, 1)

        if factor is None:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        else:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, factor)
            _, diag_1, _, _ = VBMF.EVBMF(unfold_1, factor)

        ranks = [diag_0.shape[0], diag_1.shape[1]]
        return ranks

    else:
        layer_data = tf.transpose(layer_data, [2, 0, 3, 1])
        layer_data = np.asarray(layer_data)
        layer_shape = layer_data.shape
        unfold_0 = layer_data.reshape(layer_shape[0]*layer_shape[1], -1)
        if factor is None:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        else:
            _, diag_0, _, _ = VBMF.EVBMF(unfold_0, factor)

        ranks = [diag_0.shape[0]]
        return ranks
