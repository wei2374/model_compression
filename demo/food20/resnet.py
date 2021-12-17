import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Add,\
    GlobalMaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

def ResNet(
            input_shape,
            num_classes,
            models_filename=None,
            lr=1e-3):
    feature_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = feature_extractor.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256,activation='relu')(x)
    x = Dense(128,activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(num_classes,
                        kernel_regularizer=l2(0.005), 
                        activation='softmax')(x)

    model = tf.keras.Model(inputs=feature_extractor.input, outputs=predictions)
    # model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), 
    #               loss='categorical_crossentropy', 
    #               metrics=['accuracy'])

    if models_filename is not None:
        original_model = tf.keras.models.load_model(models_filename, compile=False)
        model.load_weights(models_filename)
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    
    return model