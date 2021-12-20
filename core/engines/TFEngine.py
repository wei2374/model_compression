from core.Engine import Engine
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, DepthwiseConv2D,\
    BatchNormalization, Activation, ZeroPadding2D, Add
import numpy as np


class TFEngine(Engine):
    def get_prev_layers(self, layer):
        layers = []
        for node in layer.inbound_nodes:
            if(isinstance(node.inbound_layers, list)):
                for _layer in node.inbound_layers:
                    layers.append(_layer)
            else:
                layers.append(node.inbound_layers)
        return layers
     

    def get_next_layers(self, layer):
        layers = []
        for node in layer.outbound_nodes:
            if(isinstance(node.outbound_layer, list)):
                for _layer in node.outbound_layer:
                    layers.append(_layer)
            else:
                layers.append(node.outbound_layer)
        return layers


    def is_branch(self, layer):
        return len(self.get_next_layers(layer.raw_layer))>1

    def is_merge(self, layer):
        return len(self.get_prev_layers(layer.raw_layer))>1

    def is_type(self, layer, type):
        return isinstance(layer.raw_layer, type)

    def wrap(self, layer):
        from core.layers.BNLayer import BNLayer
        from core.layers.Conv2DLayer import Conv2DLayer
        from core.layers.DepthwiseLayer import DepthwiseLayer
        from core.layers.DenseLayer import DenseLayer
        from core.layers.FlattenLayer import FlattenLayer
        from core.Layer import Layer
        
        LAYER_MAPPING={
                    BatchNormalization: BNLayer,
                    Conv2D: Conv2DLayer,
                    DepthwiseConv2D: DepthwiseLayer,
                    Dense: DenseLayer,
                    Flatten: FlattenLayer
                }

        if(any([isinstance(layer, layer_type) for layer_type in LAYER_MAPPING])):
            for layer_type in LAYER_MAPPING:
                if(isinstance(layer, layer_type)):
                    layer_cons = LAYER_MAPPING.get(layer_type)
                    return layer_cons
        else:
            return Layer
    
    def load_model_param(self, model):
        layer_params = []
        layer_types = []
        layer_output_shape = []
        layer_bias = []
        layer_index_dic = {}

        for index, layer in enumerate(model.layers):
            layer_types.append(layer.__class__.__name__)
            layer_params.append(layer.get_weights())
            layer_index_dic[index] = layer
            layer_output_shape.append(list(layer.output_shape))
            try:
                layer_bias.append(layer.use_bias)
            except Exception:
                layer_bias.append(None)
        return np.array(layer_types), np.array(layer_params),\
            layer_output_shape, layer_bias, layer_index_dic

    def use_bias(self, layer):
        return layer.raw_layer.use_bias