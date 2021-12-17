from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_food101_data(dataset_dir,
                     bs,
                     img_hight=224,
                     img_width=224,
                     preprocessing="vgg16",
                     validation_split=0.25,
                     small_part=1):

    train_data_dir = os.path.join(dataset_dir, "train")
    validation_data_dir = os.path.join(dataset_dir, "test")
    if small_part == 1:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        train_data = train_datagen.flow_from_directory(
            str(train_data_dir),
            batch_size=bs,
            target_size=(img_hight, img_width))

    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            validation_split=1-small_part)
        train_data = train_datagen.flow_from_directory(
            str(train_data_dir),
            batch_size=bs,
            target_size=(img_hight, img_width),
            subset='training')

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    
    valid_data = test_datagen.flow_from_directory(
        str(validation_data_dir),
        target_size=(img_hight, img_width),
        batch_size=bs)

    return train_data, valid_data
