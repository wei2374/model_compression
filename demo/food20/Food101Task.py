import tensorflow as tf
from Core.Task import Task
import os
from food101 import get_food101_data
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tools.visualization.training_visualization_tool import plot_training_result
from cyclic import CyclicLR


class Food101Task(Task):
    def __init__(self, config):
        super().__init__(config)

    def prepare_model(self, type="vgg16", filename=None):
        if filename is not None:
            foldername = self.config['Model']['model_folder']
            filename = os.path.join(foldername, filename)
            model = tf.keras.models.load_model(filename)
            return model
        else:
            if type == "vgg16":
                from vgg16 import VGG16    
                model = VGG16(input_shape=[224, 224, 3], num_classes=20)
            elif type == "resnet":
                from resnet import ResNet50
                model = ResNet50(input_shape=[224, 224, 3], num_classes=20)
            elif type == "mobilenet_v2":
                from mobilenet_v2 import MobileNetV2_avg_max
                model = MobileNetV2_avg_max(input_shape=[224, 224, 3], num_classes=20)
            return model
    
    def get_dataset(self):
        ds_path = self.config['Datasets']['dataset_folder']
        train_data, valid_data = get_food101_data(
                dataset_dir=ds_path,
                bs=32)
        return train_data, valid_data
    
    def train(self, model, model_name='fine_tuned_model.h'):
        train_data, valid_data = self.get_dataset()
        foldername = self.config['Model']['model_folder']
        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)
        acc5 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='top_k_categorical_accuracy', dtype=None)

        initial_learning_rate = 1e-5
        maximal_learning_rate = 1e-3
        NUM_CLR_CYCLES = 2
        step_size = train_data.n/NUM_CLR_CYCLES/2

        opt = tf.keras.optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
        clr = CyclicLR(
            base_lr=initial_learning_rate,
            max_lr=maximal_learning_rate,
            step_size=step_size)

        model.compile(
                optimizer=opt,
                loss="categorical_crossentropy",
                metrics=[acc1, acc5])

        checkpointer = ModelCheckpoint(
                filepath=foldername,
                verbose=1,
                monitor='val_top1',
                mode='max',
                save_best_only=True)

        early_stopper = tf.keras.callbacks.EarlyStopping(
                    monitor="val_top1", patience=3, mode="max"
                )
        csv_logger = CSVLogger(csv_path)
        with tf.device('/gpu:0'):
            history = model.fit_generator(
                        train_data,
                        validation_freq=1,
                        validation_data=valid_data,
                        epochs=30,
                        verbose=1,
                        callbacks=[csv_logger, checkpointer, clr, early_stopper])

        plot_training_result(history, clr, foldername)
        model.save(os.path.join(foldername, "trained_model.h5"))