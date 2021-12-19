from abc import ABC, abstractmethod

class Engine(ABC):
    @abstractmethod
    def get_prev_layers(self, layer):
        return
    
    @abstractmethod
    def get_next_layers(self, layer):
        return
    
    @abstractmethod
    def is_branch(self, layer):
        return
    
    @abstractmethod
    def is_merge(self, layer):
        return

    @abstractmethod
    def is_type(self, layer):
        return
    
    @abstractmethod
    def use_bias(self, layer):
        return