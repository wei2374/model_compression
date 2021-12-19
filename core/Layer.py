from abc import ABC, abstractmethod
from core.engines.TFEngine import TFEngine

class Layer(ABC):
    def __init__(self, raw_layer, soft_prune=False, engine=TFEngine()):
        self.engine = engine
        self.raw_layer = raw_layer
        self.soft_prune = soft_prune

    def get_index(self, dic):
        for index, layer in dic.items():
            if layer == self.raw_layer:
                return index

    def active_prune(self, filter, model_params, layer_index_dic):
        """Call this method to adjust weights of current layer"""
        return

    def passive_prune(self, filter, model_params, layer_index_dic):
        """Call this method to be adjusted"""
        return

    def propagate(self, filter, model_params, layer_index_dic):
        """Call this method to propagate and adjust weights of other layers"""
        return

    def prune_forward(self, filter, model_params, next_layers, layer_index_dic):
        while(len(next_layers)==1 and\
            any([next_layers[0].is_type( _layer) for _layer in self.THROUGH_LAYERS])):
            for layer_type in self.THROUGH_LAYERS:
                if next_layers[0].is_type(layer_type):
                    through_layer = self.THROUGH_LAYERS.get(layer_type)(next_layers[0].raw_layer)
                    through_layer.passive_prune(filter, model_params, layer_index_dic)
                    next_layers = [through_layer]
                    break
            next_layers = next_layers[0].get_next_layers()
        return next_layers


    def pass_forward(self, next_layers):
        while(len(next_layers)==1 and\
            any([next_layers[0].is_type( _layer_type) for _layer_type in self.THROUGH_LAYERS])):
            next_layers = next_layers[0].get_next_layers()
        return next_layers


    def pass_back(self, prev_layers):
        while(len(prev_layers)==1 and\
            not prev_layers[0].is_branch() and\
            any([prev_layers[0].is_type( _layer_type) for _layer_type in self.THROUGH_LAYERS])):
            prev_layers = prev_layers[0].get_previous_layers()
        return prev_layers


    def get_previous_layers(self):
        raw_layers = self.engine.get_prev_layers(self.raw_layer)
        layers = []
        for layer in raw_layers:
            layers.append(self.wrap(layer))
        return layers


    def get_next_layers(self):
        raw_layers = self.engine.get_next_layers(self.raw_layer)
        layers = []
        for layer in raw_layers:
            layers.append(self.wrap(layer))
        return layers


    def wrap(self, raw_layer):
        return self.engine.wrap(raw_layer)(raw_layer)

    def is_type(self, type):
        return self.engine.is_type(self, type)

    def is_merge(self):
        return self.engine.is_merge(self)
    
    def is_branch(self):
        return self.engine.is_branch(self)