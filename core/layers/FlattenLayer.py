from core.Layer import Layer
import numpy as np


class FlattenLayer(Layer):
    def passive_prune(self, filter, model_params, layer_index_dic):
        layer_id = self.get_index(layer_index_dic)
        layer_output = layer_index_dic[layer_id-1]
        layer_output_shape = layer_output.output_shape
        shape = (layer_output_shape[1]*layer_output_shape[2])
        filters = []
        channels = layer_output_shape[3]
        new_filter = filter[0]
        for s in range(shape):
            filters = np.concatenate([filters, new_filter])
            new_filter = new_filter+channels
        filters = [int(i) for i in filters]
        filter = filters
        