import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add,\
    GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def MobileNetV2_avg_max(input_shape, num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(32, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation="softmax")(x)


    # y = GlobalMaxPooling2D()(base_model.output)
    # y = Dense(32, activation='relu')(y)
    # y = Dense(128, activation='relu')(y)

    # concatenated = Add()([x,y])

    # concatenated = Concatenate()([x,y])

    # concatenated = Dense(32, activation='relu')(concatenated)
    # predictions = Dense(classes, activation="softmax")(concatenated)
    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model