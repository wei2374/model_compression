from MINI.Core.Layer import Layer
import numpy as np


class DenseLayer(Layer):
    def passive_prune(self, filter, model_params, layer_index_dic):
        layer_id = self.get_index(layer_index_dic)
        if not self.soft_prune:
            model_params[layer_id][0] = np.delete(model_params[layer_id][0], filter, axis=0)
     