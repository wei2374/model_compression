from tools.visualization.training_visualization_tool import plot_training_result
import tensorflow as tf
from torch.functional import Tensor
from dataset.load_datasets import get_data_from_dataset
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from .cyclic import CyclicLR
import numpy as np
# from tools.visualization import visualize_model
from keras_flops import get_flops


def validate_model(
            dataset,
            model_path=None,
            model=None,
            bs=16,
            preprocessing="vgg16",
            split=0.25
            ):
    my_model = model if model is not None\
        else tf.keras.models.load_model(model_path, compile=False)

    _, valid_data = get_data_from_dataset(
                dataset,
                batch_size=bs,
                validation_split=split,
                preprocessing=preprocessing,
                small_part=1)

    # X_train, y_train = next(train_data)
    # for i in range(3):
    #     img, label = next(train_data)
    #     X_train = np.append(X_train, img, axis=0)
    #     y_train = np.append(y_train, label, axis=0)

    acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
        k=1, name="top1", dtype=None)
    acc5 = tf.keras.metrics.TopKCategoricalAccuracy(
        k=5, name="top5", dtype=None)

    my_model.compile(metrics=[acc1, acc5])
    flops = get_flops(my_model, batch_size=1)
    print(flops)
    with tf.device('/gpu:0'):
        history = my_model.evaluate(valid_data, verbose=1)

    from tensorflow.keras.backend import count_params
    params = int(
            np.sum([count_params(p) for p in my_model.trainable_weights])
            ) + int(
            np.sum([count_params(p) for p in my_model.non_trainable_weights]))

    return history[1:], flops, params
    # if pruned_ratio is not None:
    #     visualize_model(my_model, pruned_ratio)


def train_model(
        raw_model,
        method,
        dataset,
        foldername,
        train_data=None,
        valid_data=None,
        bs=16,
        epoch_n=20,
        preprocessing="vgg16",
        small_part=0.1,
        validation_freq=1):

    csv_path = foldername+'/logfile.log'
    model_path = foldername+"/compressed_model.h5"

    if method == "cycling":
        train_data, valid_data = [train_data, valid_data]
        if train_data is None:
            train_data, valid_data = get_data_from_dataset(
                dataset,
                batch_size=bs,
                preprocessing=preprocessing,
                small_part=small_part)

        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)
        acc5 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='top_k_categorical_accuracy', dtype=None)

        initial_learning_rate = 1e-5
        maximal_learning_rate = 1e-3
        step_size = train_data.n/2
        opt = SGD(lr=initial_learning_rate, momentum=0.9)
        clr = CyclicLR(
            base_lr=initial_learning_rate,
            max_lr=maximal_learning_rate,
            step_size=step_size)

        raw_model.compile(
                optimizer=opt,
                loss="categorical_crossentropy",
                metrics=[acc1, acc5])

        checkpointer = ModelCheckpoint(
                filepath=model_path,
                verbose=1,
                monitor='val_top1',
                mode='max',
                save_best_only=True)

        early_stopper = tf.keras.callbacks.EarlyStopping(
                    monitor="val_top1", patience=1, mode="max"
                )
        csv_logger = CSVLogger(csv_path)
        with tf.device('/gpu:0'):
            history = raw_model.fit_generator(
                        train_data,
                        validation_freq=validation_freq,
                        validation_data=valid_data,
                        epochs=epoch_n,
                        verbose=1,
                        callbacks=[csv_logger, checkpointer, clr, early_stopper])

        plot_training_result(history, clr, foldername)

    elif method == "adam":
        train_data, valid_data = [train_data, valid_data]

        if train_data is None:
            train_data, valid_data = get_data_from_dataset(
                dataset,
                batch_size=bs,
                preprocessing=preprocessing,
                small_part=small_part)

        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)
        acc5 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='top_k_categorical_accuracy', dtype=None
        )
        checkpointer = ModelCheckpoint(
                        filepath=model_path,
                        verbose=1,
                        monitor='val_top1',
                        mode='max',
                        save_best_only=True)

        with tf.device('/gpu:0'):
            raw_model.compile(
                optimizer='adam',
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[acc1, acc5])
            history = raw_model.fit_generator(
                        train_data,
                        epochs=10,
                        validation_data=valid_data,
                        verbose=1,
                        callbacks=[checkpointer])
            plot_training_result(history, None, foldername)

    elif method == "default":
        train_data, valid_data = [train_data, valid_data]

        if train_data is None:
            train_data, valid_data = get_data_from_dataset(
                dataset,
                batch_size=bs,
                preprocessing=preprocessing,
                small_part=small_part)

        acc1 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=1, name="top1", dtype=None)
        acc5 = tf.keras.metrics.TopKCategoricalAccuracy(
            k=5, name='top_k_categorical_accuracy', dtype=None
        )
        checkpointer = ModelCheckpoint(
                        filepath=model_path,
                        verbose=1,
                        monitor='val_top1',
                        mode='max',
                        save_best_only=True)

        with tf.device('/gpu:0'):
            sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
            raw_model.compile(
                optimizer=sgd,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[acc1, acc5])
            history = raw_model.fit_generator(
                train_data,
                epochs=3,
                validation_data=valid_data,
                validation_freq=1,
                verbose=2,
                callbacks=[checkpointer])

            sgd = tf.keras.optimizers.SGD(learning_rate=0.0001)
            raw_model.compile(
                optimizer=sgd,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[acc1, acc5])
            history = raw_model.fit_generator(
                train_data,
                epochs=3,
                validation_data=valid_data,
                validation_freq=1,
                verbose=2,
                callbacks=[checkpointer])

            sgd = tf.keras.optimizers.SGD(learning_rate=0.00001)
            raw_model.compile(
                optimizer=sgd,
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[acc1, acc5])
            history = raw_model.fit_generator(
                train_data,
                epochs=3,
                validation_data=valid_data,
                validation_freq=1,
                verbose=2,
                callbacks=[checkpointer])
        plot_training_result(history, None, foldername)

    return history, train_data, valid_data