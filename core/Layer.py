from abc import ABC, abstractmethod
from tensorflow.python.keras.layers.convolutional import Conv
from compression_tools.pruning.helper_functions import get_layer_index
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, DepthwiseConv2D,\
    BatchNormalization, Activation, ZeroPadding2D, Add
        

class Layer(ABC):
    def __init__(self, layer, dic, soft_prune=False):
        self.index = get_layer_index(dic, layer)
        self.layer = layer
        self.prunable = False
        self.soft_prune = soft_prune
        self.dic = dic

    def active_prune(self, filter, model_params):
        """Call this method to adjust weights of current layer"""
        return

    def passive_prune(self, filter, model_params):
        """Call this method to be adjusted"""
        return

    def propagate(self, filter, model_params):
        """Call this method to propagate and adjust weights of other layers"""
        return

    def pass_through(self, filter, model_params):
        return

    def get_previous_layers(self):
        layers = []
        for node in self.layer.inbound_nodes:
            if(isinstance(node.inbound_layers, list)):
                for _layer in node.inbound_layers:
                    layers.append(self.wrap(_layer))
            else:
                layers.append(self.wrap(node.inbound_layers))

        return layers


    def get_next_layers(self):
        layers = []
        for node in self.layer.outbound_nodes:
            if(isinstance(node.outbound_layer, list)):
                for _layer in node.outbound_layer:
                    layers.append(self.wrap(_layer))
                    
            else:
                layers.append(self.wrap(node.outbound_layer))
                    
        return layers

    def is_branch(self):
        return self.get_next_layers()>1

    def wrap(self, layer):
        from core.layers.BNLayer import BNLayer
        from core.layers.Conv2DLayer import Conv2DLayer
        from core.layers.DepthwiseLayer import DepthwiseLayer
        from core.layers.DenseLayer import DenseLayer
        from core.layers.FlattenLayer import FlattenLayer
        
        
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
                    return layer_cons(layer, self.dic)
        else:
            return Layer(layer, self.dic)
