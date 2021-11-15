from .food101 import get_food101_data
from .mnist import GetData
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_from_dataset(
            dataset,
            batch_size=32,
            validation_split=0.25,
            small_part=1,
            preprocessing="vgg16"
            ):
    '''
    Call this function to get training data and validation data
    for model training
    '''
    if 'food_20' in dataset or 'food_101' in dataset:
        train_data, valid_data = get_food101_data(
                dataset_dir=dataset,
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
