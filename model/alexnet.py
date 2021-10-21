import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout


def AlexNet(input_shape, 
                 num_classes,
                 lr=1e-3,
                 bn=False,
                 models_filename=None,
                 s=0.001):
        """Init AlexNet V2 model.

        Args:
            input_shape: (tuple) image shape: (hight, width, channels)
            num_classes: (int) number of classes.
            lr: (float) learning rate.
            bn: add batch normalization layer
        """
        img_input = tf.keras.layers.Input(shape=input_shape)
        x = (tf.keras.layers.Conv2D(96, kernel_size=(3,3), strides= 4,
                        input_shape= input_shape)(img_input))
        if bn:
            x = (tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x))
        x = (tf.keras.layers.Activation(tf.keras.activations.relu))(x)
        x = (tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides= (2,2),
                              data_format= None))(x)

        x = (tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'same'))(x)
        if bn:
            x = (tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x))
        x = (tf.keras.layers.Activation(tf.keras.activations.relu))(x)
        x = (tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides= (2,2),
                               data_format= None))(x) 

        x = (tf.keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same'))(x)
        if bn:
            x = (tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x))
        x = (tf.keras.layers.Activation(tf.keras.activations.relu))(x)

        x = (tf.keras.layers.Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'same'))(x)
        if bn:
            x = (tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x))
        x = (tf.keras.layers.Activation(tf.keras.activations.relu))(x)

        x = (tf.keras.layers.Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'same'))(x)
        if bn:
            x = (tf.keras.layers.BatchNormalization(gamma_regularizer=tf.keras.regularizers.l1(s))(x))
        x = (tf.keras.layers.Activation(tf.keras.activations.relu))(x)

        x = (tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides= (2,2),
                               data_format= None))(x)

        x = (tf.keras.layers.Flatten())(x)
        x = (tf.keras.layers.Dense(4096, activation= 'relu'))(x)
        x = (Dropout(0.4))(x)
        x = (tf.keras.layers.Dense(4096, activation= 'relu'))(x)
        x = (Dropout(0.4))(x)
        #x = (tf.keras.layers.Dense(1000, activation= 'relu'))(x)
        x = (tf.keras.layers.Dense(num_classes, activation= 'softmax'))(x)
        #x = (tf.keras.layers.Dense(num_classes, activation='softmax'))(x)
        model = tf.keras.models.Model(inputs=img_input, outputs=x)
        
        model.compile(optimizer="adam",  #tf.keras.optimizers.Adam(0.001),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
        if models_filename is not None:
            model.load_weights(models_filename)
        
        return model
        #print(self.summary())
