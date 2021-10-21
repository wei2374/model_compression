import tensorflow as tf
from models.vgg16 import VGG16


class ModelLoader:
    '''
    This class deals with all functions related with getting model
    '''
    def __init__(self, model_type, input_shape, classes_num, model_filename=None):
        if model_type == "vgg16":
            self.my_model = VGG16(
                    input_shape=input_shape,
                    num_classes=classes_num,
                    models_filename=model_filename
                )
        else:
            self.my_model = tf.keras.models.load_model(
                        model_filename,
                        compile=False)

    def get_model(self):
        return self.my_model
