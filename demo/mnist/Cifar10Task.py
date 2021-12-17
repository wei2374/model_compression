import tensorflow as tf
from core.Task import Task
import os
import numpy as np
from cifar10 import GetData
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tools.visualization.training_visualization_tool import plot_training_result
from tensorflow.keras.callbacks import LearningRateScheduler

class SpeechCmdTask(Task):
    def __init__(self, config):
        super().__init__(config)

    def prepare_model(self, filename=None):
        if filename is not None:
            foldername = self.config['Model']['model_folder']
            filename = os.path.join(foldername, filename)
            model = tf.keras.models.load_model(filename)
        else:   
            from resnet_tinyml import resnet_v1_eembc
            model = resnet_v1_eembc(
                input_shape=[32, 32, 3], num_classes=10)
        return model
    
    def get_dataset(self):
        test_data, train_data, test_labels, train_labels, shape = GetData()
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(train_data)
        test_datagen = ImageDataGenerator()
        train_data = datagen.flow(train_data, train_labels, batch_size=40)
        valid_data = test_datagen.flow(test_data, test_labels, batch_size=40)

        return train_data, valid_data


    def train(self, model, model_name='fine_tuned_model.h'):
        def lr_schedule(epoch):
            initial_learning_rate = 0.001
            decay_per_epoch = 0.99
            lrate = initial_learning_rate * (decay_per_epoch ** epoch)
            print('Learning rate = %f'%lrate)
            return lrate

        epoch_n = 100
        train_data, valid_data = self.get_dataset()
        foldername = self.config['Model']['model_folder']
        checkpointer = ModelCheckpoint(
                        filepath=foldername,
                        verbose=1,
                        monitor='val_top1',
                        mode='max',
                        save_best_only=True)
        optimizer = tf.keras.optimizers.Adam()
        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)

        lr_scheduler = LearningRateScheduler(lr_schedule)
        model.compile(
            optimizer=optimizer, loss='categorical_crossentropy', metrics=[acc1] )

        history = model.fit(train_data, validation_data=valid_data,
              epochs=epoch_n, callbacks=[lr_scheduler, checkpointer])
        
        plot_training_result(history, None, foldername)
        model.save(os.path.join(foldername, "trained_model.h5"))
    