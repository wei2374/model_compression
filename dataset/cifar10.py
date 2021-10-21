import numpy as np
import tensorflow as tf
from pathlib2 import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_cifar10_data(dataset_dir,  
                     batch_size,
                     img_hight=32,
                     img_width=32,
                     preprocessing="vgg16"):
                     
    num_classes = 10
    data_dir = Path(dataset_dir + "/train")
    test_data_dir = Path(dataset_dir + "/test")
    # Prepare train data generator with necessary augmentations 
    # and validation split
    
    if preprocessing == "vgg16":
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = False,
            fill_mode = "nearest",
            zoom_range = 0,
            width_shift_range = 0,
            height_shift_range=0,
            rotation_range=0)

    elif preprocessing == "resnet50":
        train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                brightness_range=None,
                shear_range=0.2,
                zoom_range=0.2,
                channel_shift_range=0.0,
                fill_mode="nearest",
                horizontal_flip=True,
                vertical_flip=True,
                preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                validation_split=0.25
            )

    # Generate training data
    print('Train Data')
    train_data = train_datagen.flow_from_directory(
        str(data_dir),
        batch_size=batch_size,
        shuffle=True,
        target_size=(img_hight, img_width),
        class_mode = "categorical")
    
    valid_data = train_datagen.flow_from_directory(
        str(test_data_dir),
        batch_size=batch_size,
        shuffle=True,
        target_size=(img_hight, img_width),
        class_mode = "categorical")

    return train_data, valid_data, num_classes