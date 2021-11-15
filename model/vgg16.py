import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Activation,\
    BatchNormalization
from tensorflow.keras import activations
import numpy as np


def VGG16(
        input_shape,
        num_classes,
        models_filename=None,
        bn=False,
        s=0.0001):

    """Init VGG16 model.

    Args:
        input_shape: (tuple) image shape: (hight, width, channels)
        num_classes: (int) number of classes.
        lr: (float) learning rate.
        models_filename : pretrained model with same architecture
        bn: if BatchNormalization layer is inserted between Conv2D and Activation layer
    """
    initializer = 'HeUniform'
    # initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    img_input = Input(shape=input_shape)
    x = (Conv2D(
        filters=64, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(img_input)
    # if bn:
    #     x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x))

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer=initializer)(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x))

    x = (Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=256, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x))

    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x))

    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (Conv2D(filters=512, kernel_size=(3, 3), padding="same", kernel_initializer=initializer))(x)
    if bn:
        x = BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x)
    x = Activation(activations.relu)(x)
    x = (MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x))

    x = (Flatten())(x)
    x = (Dense(101*2, activation="relu"))(x)
    x = (Dense(101*2, activation="relu"))(x)
    x = (Dense(num_classes, activation="softmax"))(x)

    model = tf.keras.models.Model(inputs=img_input, outputs=x)
    model.build(input_shape=input_shape)
    if models_filename is not None:
        if models_filename == "original_vgg":
            original_model = tf.keras.applications.VGG16(
                                weights="imagenet")
        else:
            original_model = tf.keras.models.load_model(models_filename, compile=False)
        if(original_model.output_shape[1] == num_classes and not bn):
            model.load_weights(models_filename)
        else:
            layer_weights = []
            for layer in original_model.layers:
                if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
                    layer_weights.append(layer.weights)
            counter = 0
            for layer in model.layers:
                if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
                    W = np.asarray(layer_weights[counter][0])
                    bias = np.asarray(layer_weights[counter][1])
                    layer.set_weights([W, bias])
                    counter += 1

    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    return model
