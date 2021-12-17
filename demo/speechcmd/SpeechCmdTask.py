import tensorflow as tf
from core.Task import Task
import os
import numpy as np
from keras_flops import get_flops
from dataset import kws_util
from dataset.get_dataset import get_training_data


class SpeechCmdTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def prepare_model(self, filename=None):
        if filename is not None:
            foldername = self.config['Model']['model_folder']
            filename = os.path.join(foldername, filename)
            model = tf.keras.models.load_model(filename)
        else:   
            from dscnn_tinyml import DSCNN
            model = DSCNN(input_shape=[49, 10, 1], num_classes=12)
        return model
    
    def get_dataset(self):
        Flags, unparsed = kws_util.parse_command()
        train_data, ds_test, valid_data = get_training_data(Flags)
        train_data = train_data.shuffle(85511)
        valid_data = valid_data.shuffle(10102)
        return train_data, valid_data
    
    def train(self, model, model_name='fine_tuned_model.h'):
        train_data, valid_data = self.get_dataset()
        Flags, unparsed = kws_util.parse_command()
        callbacks = kws_util.get_callbacks(args=Flags)
        history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=Flags.epochs,
                    callbacks=callbacks)
        
        foldername = self.config['Model']['model_folder']
        model.save(os.path.join(foldername, "trained_model.h5"))
