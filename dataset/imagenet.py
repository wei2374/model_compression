import numpy as np
from pathlib2 import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_imagenet_data(dataset_dir,
                      batch_size,
                      img_hight=224,
                      img_width=224):

    num_classes = 1000
    data_dir = Path(dataset_dir + "/train")

    # Prepare train data generator with necessary augmentations
    train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.25)
    # Generate training data
    print('Train Data')
    train_data = train_datagen.flow_from_directory(
        str(data_dir),
        batch_size=batch_size,
        shuffle=True,
        target_size=(img_hight, img_width),
        subset='training')
    # Generate validation data
    print('\nValidation Data')
    valid_data = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=(img_hight, img_width),
        batch_size=batch_size,
        subset='validation')
    return train_data, valid_data, num_classes
