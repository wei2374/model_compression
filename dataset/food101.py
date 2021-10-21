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

    if dataset_dir == "/home/wei-bshg/Documents/datasets/food-101" or\
            dataset_dir == "/home/wei-bshg/Documents/datasets/food-20":
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
                # train_datagen = ImageDataGenerator(
                #     rescale=1./255,
                #     horizontal_flip=False,
                #     fill_mode="nearest",
                #     zoom_range=0,
                #     width_shift_range=0,
                #     height_shift_range=0,
                #     rotation_range=0,
                #     validation_split=validation_split)

                # test_datagen = ImageDataGenerator()

                # other_data = train_datagen.flow_from_directory(
                #         str(train_data_dir),
                #         batch_size=bs,
                #         shuffle=False,
                #         target_size=(img_hight, img_width),
                #         subset='training')

                # valid_data = train_datagen.flow_from_directory(
                #     str(train_data_dir),
                #     target_size=(img_hight, img_width),
                #     batch_size=bs,
                #     shuffle=True,
                #     subset='validation')

                # if small_part < 1:
                #     print(f"Generating {small_part} of training data")
                #     X_train, y_train = next(other_data)
                #     num = other_data.n + valid_data.n
                #     mini_size = int(small_part*(num)/bs)-1
                #     for i in range(mini_size):
                #         img, label = next(other_data)
                #         X_train = np.append(X_train, img, axis=0)
                #         y_train = np.append(y_train, label, axis=0)

                #     train_data = test_datagen.flow(X_train, y_train, batch_size=bs, shuffle=False)
                # else:
                #     train_data = other_data

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
                # train_datagen = ImageDataGenerator(
                #             rotation_range=40,
                #             width_shift_range=0.2,
                #             height_shift_range=0.2,
                #             rescale=1./255,
                #             zoom_range=0.2,
                #             horizontal_flip=True,
                #             fill_mode='nearest',
                #             validation_split=validation_split)

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

    # elif dataset_dir == "/home/wei-bshg/Documents/datasets/food-30":
    #     if preprocessing == "vgg16":
    #         train_datagen = ImageDataGenerator(
    #             rescale=1./255,
    #             horizontal_flip=False,
    #             fill_mode="nearest",
    #             zoom_range=0,
    #             width_shift_range=0,
    #             height_shift_range=0,
    #             rotation_range=0,
    #             validation_split=validation_split)
    #     else:
    #         train_datagen = ImageDataGenerator(
    #                 rotation_range=20,
    #                 width_shift_range=0.2,
    #                 height_shift_range=0.2,
    #                 brightness_range=None,
    #                 shear_range=0.2,
    #                 zoom_range=0.2,
    #                 channel_shift_range=0.0,
    #                 fill_mode="nearest",
    #                 horizontal_flip=True,
    #                 vertical_flip=True,
    #                 preprocessing_function=preprocess_input,
    #                 validation_split=validation_split
    #             )

    #     test_datagen = ImageDataGenerator()

    #     other_data = train_datagen.flow_from_directory(
    #             str(train_data_dir),
    #             batch_size=bs,
    #             shuffle=True,
    #             target_size=(img_hight, img_width),
    #             subset='training')

    #     valid_data = train_datagen.flow_from_directory(
    #         str(train_data_dir),
    #         target_size=(img_hight, img_width),
    #         batch_size=bs,
    #         shuffle=True,
    #         subset='validation')

    #     if small_part < 1:
    #         print(f"Generating {small_part} of training data")
    #         X_train, y_train = next(other_data)
    #         num = other_data.n + valid_data.n
    #         mini_size = int(small_part*(num)/bs)-1
    #         for i in range(mini_size):
    #             img, label = next(other_data)
    #             X_train = np.append(X_train, img, axis=0)
    #             y_train = np.append(y_train, label, axis=0)

    #         train_data = test_datagen.flow(X_train, y_train, batch_size=bs)
    #     else:
    #         train_data = other_data

    return train_data, valid_data
