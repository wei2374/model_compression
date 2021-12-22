from abc import ABC, abstractmethod
import numpy as np

class Evaluator(ABC):
    def __init__(self, engine, task):
        self.engine = engine
        self.task = task

    @abstractmethod
    def evaluate(self, model):
        return
    
    def normalize(self, crits):
        for layer_index in crits:
            crits_norm = np.linalg.norm(crits[layer_index])
            crits[layer_index] = crits[layer_index]/crits_norm