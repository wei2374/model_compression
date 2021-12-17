from core.Layer import Layer
import numpy as np


class BNLayer(Layer):
    def passive_prune(self, filter, model_params):
        if not self.soft_prune:
            model_params[self.index][0] = np.delete(model_params[self.index][0], filter)
            model_params[self.index][1] = np.delete(model_params[self.index][1], filter)
            model_params[self.index][2] = np.delete(model_params[self.index][2], filter)
            model_params[self.index][3] = np.delete(model_params[self.index][3], filter) 
