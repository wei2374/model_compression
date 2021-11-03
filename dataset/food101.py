import numpy as np
from pathlib2 import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


def get_food101_data(dataset_dir,
                     bs,
                     img_hight=224,
                     img_width=224,
                     preprocessing="vgg16",
                     validation_split=0.25,
                     small_part=1):

    train_data_dir = Path(dataset_dir + "/images")

    if dataset_dir == "/home/wei-bshg/Documents/datasets/food-20":
        if preprocessing is "vgg16":
            if small_part == 1:
                train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        horizontal_flip=False,
                        fill_mode="nearest",
                        zoom_range=0,
                        width_shift_range=0,
                        height_shift_range=0,
                        rotation_range=0,
                        validation_split=validation_split)

                train_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    batch_size=bs,
                    # shuffle=False,
                    target_size=(img_hight, img_width),
                    subset='training')

                valid_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    target_size=(img_hight, img_width),
                    batch_size=bs,
                    subset='validation')
            else:
                train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        horizontal_flip=False,
                        fill_mode="nearest",
                        zoom_range=0,
                        width_shift_range=0,
                        height_shift_range=0,
                        rotation_range=0,
                        validation_split=(1-small_part))

                train_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    batch_size=bs,
                    # shuffle=False,
                    target_size=(img_hight, img_width),
                    subset='training')

                train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        horizontal_flip=False,
                        fill_mode="nearest",
                        zoom_range=0,
                        width_shift_range=0,
                        height_shift_range=0,
                        rotation_range=0,
                        validation_split=validation_split)

                valid_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    target_size=(img_hight, img_width),
                    batch_size=bs,
                    subset='validation')

        elif preprocessing is "resnet50":
            if small_part == 1:
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
                    preprocessing_function=preprocess_input,
                    validation_split=validation_split
                )

                train_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    batch_size=bs,
                    shuffle=True,
                    target_size=(img_hight, img_width),
                    subset='training')

                valid_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    target_size=(img_hight, img_width),
                    batch_size=bs,
                    subset='validation')
            else:
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
                    preprocessing_function=preprocess_input,
                    validation_split=(1-small_part)
                )
                train_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    batch_size=bs,
                    shuffle=True,
                    target_size=(img_hight, img_width),
                    subset='training')

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
                    preprocessing_function=preprocess_input,
                    validation_split=validation_split
                )
                valid_data = train_datagen.flow_from_directory(
                    str(train_data_dir),
                    target_size=(img_hight, img_width),
                    batch_size=bs,
                    subset='validation')
        elif preprocessing is "mv":
            train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # test_datagen = ImageDataGenerator(rescale=1. / 255)

        # train_data = train_datagen.flow_from_directory(
        #     str(train_data_dir),
        #     target_size=(img_hight, img_width),
        #     batch_size=bs,
        #     class_mode='categorical')

        # valid_data = test_datagen.flow_from_directory(
        #     str(train_data_dir),
        #     target_size=(img_hight, img_width),
        #     batch_size=bs,
        #     class_mode='categorical')

    elif dataset_dir == "/home/wei-bshg/Documents/datasets/food-101":
        train_data_dir = "/home/wei-bshg/Documents/datasets/food-101train"
        validation_data_dir = "/home/wei-bshg/Documents/datasets/food-101test"

        # train_datagen = ImageDataGenerator(
        #                 rescale=1./255,
        #                 horizontal_flip=False,
        #                 fill_mode="nearest",
        #                 zoom_range=0,
        #                 width_shift_range=0,
        #                 height_shift_range=0,
        #                 rotation_range=0)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_data = train_datagen.flow_from_directory(
            str(train_data_dir),
            batch_size=bs,
            target_size=(img_hight, img_width))

        valid_data = test_datagen.flow_from_directory(
            str(validation_data_dir),
            target_size=(img_hight, img_width),
            batch_size=bs)


    return train_data, valid_data
