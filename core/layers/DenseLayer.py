from core.Layer import Layer
import numpy as np


class DenseLayer(Layer):
    def passive_prune(self, filter, model_params):
        if not self.soft_prune:
            model_params[self.index][0] = np.delete(model_params[self.index][0], filter, axis=0)
     