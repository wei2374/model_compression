from abc import ABC, abstractmethod

class CompressStrategy(ABC):
    @abstractmethod
    def param_estimation(self, model):
        return

    @abstractmethod
    def self_compress(self, layer, param):
        return

    @abstractmethod
    def reconstruct_model(self, layer, param, model):
        return