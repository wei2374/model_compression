dataset_name = "VisualWakeWords"
dataset_train_zip = "train2014.zip"
dataset_val_zip = "val2014.zip"
dataset_url = "http://images.cocodataset.org/zips/"
dataset_annotations_zip = "annotations_trainval2014.zip"
dataset_annotations_url = "http://images.cocodataset.org/annotations/"
TARGET_SIZE = 96
import os


def GetData():
    # create keras dataset from directory
    import tensorflow as tf
    output_dir = "/home/wei/Documents/workspace/TensorNAS/TensorNAS/Demos/Datasets/tmp/VisualWakeWords"
    BATCH_SIZE = 32
    validation_split = 0.1

    train_dir = os.path.join(output_dir, "train_dataset")
    val_dir = os.path.join(output_dir, "test_dataset")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1.0 / 255,
    )
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        color_mode="rgb",
    )

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        color_mode="rgb",
    )
    print(train_generator.class_indices)

    return train_generator, val_generator, train_generator.image_shape

