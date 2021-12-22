from abc import ABC, abstractmethod
import os
import configparser
from keras_flops import get_flops
import numpy as np
import tensorflow as tf
from MINI.Strategies.LayerPruning import LayerPruning

class Task(ABC):
    def __init__(self, cfg):
        self.strategy = LayerPruning(self)
        self.config = configparser.ConfigParser()
        p = os.path.abspath('.')
        config_path = os.path.join(p, cfg)
        self.config.read(config_path)
        model_dir = self.config['Model']['model_folder']
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    @abstractmethod
    def prepare_model(self, filename=None):
        """Call this method to prepare original model"""
        return

    @abstractmethod
    def get_dataset(self):
        """Call this method to get dataset"""
        return

    @abstractmethod
    def train(self):
        """Call this method to train model"""
        return

    def evaluate(self, model):
        _, valid_data = self.get_dataset()
        flops = get_flops(model, batch_size=1)
        print(flops)
        with tf.device('/gpu:0'):
            history = model.evaluate(valid_data, verbose=1)

        from tensorflow.keras.backend import count_params
        params = int(
                np.sum([count_params(p) for p in model.trainable_weights])
                ) + int(
                np.sum([count_params(p) for p in model.non_trainable_weights]))

        return history[1:], flops, params


    def compress(self, model):
        compressed_model = self.strategy.run(model, self.config)
        return compressed_model
        # from compression_tools.pruning.helper_functions import load_model_param
        # _, _, _, _, layer_index_dic = load_model_param(model)
