from abc import ABC, abstractmethod
import numpy as np

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, layer):
        return
    
    def normalize(self, crits):
        for layer_index in crits:
            crits_norm = np.linalg.norm(crits[layer_index])
            crits[layer_index] = crits[layer_index]/crits_norm