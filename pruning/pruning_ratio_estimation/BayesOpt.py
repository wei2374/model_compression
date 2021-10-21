import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorly.decomposition import partial_tucker


class BayesOpt_rank_selection:
    def __init__(self, layer):
        self.layer = layer

    def f(self, x):
        layer = self.layer
        x1 = x[:, 0]
        x2 = x[:, 1]

        ranks = [int(x1), int(x2)]

        weights = np.asarray(layer.get_weights()[0])
        # bias = layer.get_weights()[1]
        layer_data = tl.tensor(weights)
        layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

        core, [last, first] = partial_tucker(
            layer_data, modes=[0, 1], rank=ranks, init="svd"
        )

        recon_error = tl.norm(
            layer_data - tl.tucker_to_tensor((core, [last, first])),
            2,
        ) / tl.norm(layer_data, 2)

        # recon_error = np.nan_to_num(recon_error)

        ori_out = layer_data.shape[0]
        ori_in = layer_data.shape[1]
        ori_ker = layer_data.shape[2]
        ori_ker2 = layer_data.shape[3]

        first_out = first.shape[0]
        first_in = first.shape[1]

        core_out = core.shape[0]
        core_in = core.shape[1]

        last_out = last.shape[0]
        last_in = last.shape[1]

        original_computation = ori_out * ori_in * ori_ker * ori_ker2
        decomposed_computation = (
            (first_out * first_in)
            + (core_in * core_out * ori_ker * ori_ker2)
            + (last_in * last_out)
        )

        computation_error = 0.8 * decomposed_computation / original_computation

        if computation_error > 1.0:
            computation_error = 5.0

        Error = float(recon_error + computation_error)

        print("%d, %d, %f, %f, %f" % (x1, x2, recon_error, computation_error, Error))

        return Error
