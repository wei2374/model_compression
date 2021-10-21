from .food101 import get_food101_data
from .mnist import GetData
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:
    '''
    This class deals with all functions related with getting data
    '''
    def __init__(self, dataset, batch_size, val_split, small_part, preprocessing):
        self.train_data, self.valid_data = get_food101_data(
                dataset_dir=dataset,
                bs=batch_size,
                preprocessing=preprocessing,
                validation_split=val_split,
                small_part=small_part)

    def get_all_data(self):
        return self.train_data, self.valid_data

    def get_valid_data(self):
        return self.valid_data


def get_data_from_dataset(
            dataset,
            batch_size=16,
            validation_split=0.25,
            small_part=1,
            preprocessing="vgg16"
            ):
    '''
    Call this function to get training data and validation data
    for model training
    '''
    dataset_dir = "/home/wei-bshg/Documents/datasets/"
    if dataset == "food101":
        train_data, valid_data = get_food101_data(
                dataset_dir=dataset_dir+"food-101",
                bs=batch_size,
                preprocessing=preprocessing,
                validation_split=validation_split,
                small_part=small_part)

    if dataset == "food20":
        train_data, valid_data = get_food101_data(
                dataset_dir=dataset_dir+"food-20",
                bs=batch_size,
                preprocessing=preprocessing,
                validation_split=validation_split,
                small_part=small_part)

    elif dataset == "mnist":
        images_train, images_test, labels_train, labels_test, _ = GetData()
        test_datagen = ImageDataGenerator()
        train_data = test_datagen.flow(images_train, labels_train)
        valid_data = test_datagen.flow(images_test, labels_test)

    return train_data, valid_data
