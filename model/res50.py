import tensorflow as tf

def ResNet50(input_shape, 
                 num_classes,
                 models_filename=None,
                 lr=1e-3):

        """Init ResNet50 model.

        Args:
            input_shape: (tuple) image shape: (hight, width, channels)
            num_classes: (int) number of classes.
            lr: (float) learning rate.
            models_filename : pretrained model with same architecture
        """
        dropout = 0.2
        base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        features = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(4096, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        
        if models_filename is not None:
            model.load_weights(models_filename)
        
        return model
        #model.summary()        