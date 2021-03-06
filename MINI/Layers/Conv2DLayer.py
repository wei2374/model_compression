from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, DepthwiseConv2D,\
    BatchNormalization, ZeroPadding2D, Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D

from MINI.Core.Layer import Layer
from MINI.Layers.BNLayer import BNLayer
from MINI.Layers.DenseLayer import DenseLayer
from MINI.Layers.FlattenLayer import FlattenLayer
from MINI.Layers.DenseLayer import DenseLayer
from MINI.Layers.DepthwiseLayer import DepthwiseLayer
from MINI.Layers.FlattenLayer import FlattenLayer

class Conv2DLayer(Layer):
    THROUGH_LAYERS = {
        BatchNormalization: BNLayer,
        DepthwiseConv2D: DepthwiseLayer,
        Flatten: FlattenLayer,
        ZeroPadding2D: Layer,
        Activation: Layer,
        ReLU: Layer,
        Dropout: Layer,
        AveragePooling2D: Layer,
        GlobalAveragePooling2D: Layer,
        MaxPooling2D: Layer,
    }

    STOP_LAYERS = {
        Conv2D: 'Conv2DLayer',
        Dense: DenseLayer,
    }

    ACT_LAYERS = {
        Activation: Layer,
        ReLU: Layer,
    }

    def active_prune(self, filter, model_params, layer_index_dic):
        layer_id = self.get_index(layer_index_dic)
        if not self.soft_prune:
            model_params[layer_id][0] = np.delete(model_params[layer_id][0], filter, axis=3)
            if self.engine.use_bias(self):
                model_params[layer_id][1] = np.delete(model_params[layer_id][1], filter, axis=0)
        else:
            model_params[layer_id][0][:, :, :, filter] = float(0)
            if self.engine.use_bias(self):
                model_params[layer_id][1][filter] = float(0)


    def passive_prune(self, filter, model_params, layer_index_dic):
        layer_id = self.get_index(layer_index_dic)
        if not self.soft_prune:
            model_params[layer_id][0] = np.delete(model_params[layer_id][0], filter, axis=2)
