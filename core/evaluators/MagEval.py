import numpy as np
from core.Evaluator import Evaluator
from tensorflow.keras.layers import Conv2D


class MagEval(Evaluator):
    def evaluate(self, layer):
        if(isinstance(layer, Conv2D)):
            layer_weights = np.asarray(layer.get_weights()[0])
            channels = layer_weights.shape[3]
            channel_weights = layer_weights.transpose(3, 0, 1, 2).reshape(channels, -1)
            channel_weights_abs_av = np.average(np.abs(channel_weights), axis=1)
            return channel_weights_abs_av
