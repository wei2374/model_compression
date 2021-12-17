import tensorflow as tf
from core.Task import Task
import os
import numpy as np
from keras_flops import get_flops
from vww import GetData
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tools.visualization.training_visualization_tool import plot_training_result


class VwwTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def prepare_model(self, filename=None):
        foldername = self.config['Model']['model_folder']
        if filename is not None:
            foldername = self.config['Model']['model_folder']
            filename = os.path.join(foldername, filename)
            model = tf.keras.models.load_model(filename)
        else:
            from mobilenet_tinyml import MobileNet_tinyml
            model = model = MobileNet_tinyml(input_shape=[96, 96, 3], num_classes=2)
        return model
    
    def get_dataset(self):
        train_data, valid_data, shape = GetData()
        return train_data, valid_data
    
    def train(self, model, model_name='fine_tuned_model.h'):
        train_data, valid_data = self.get_dataset()
        foldername = self.config['Model']['model_folder']
        checkpointer = ModelCheckpoint(
                        filepath=foldername,
                        verbose=1,
                        monitor='val_top1',
                        mode='max',
                        save_best_only=True)
        
        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)
        
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc1] )

        history = model.fit(train_data, validation_data=valid_data,
              epochs=20, callbacks=[checkpointer])
        
        optimizer = tf.keras.optimizers.Adam(0.0005)
        model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc1] )

        history = model.fit(train_data, validation_data=valid_data,
              epochs=10, callbacks=[checkpointer])

        optimizer = tf.keras.optimizers.Adam(0.00025)
        model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc1] )

        history = model.fit(train_data, validation_data=valid_data,
              epochs=20, callbacks=[checkpointer])
        
        # plot_training_result(history, None, foldername)
        model.save(os.path.join(foldername, "trained_model.h5"))
