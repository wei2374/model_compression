from tensorflow.keras.utils import to_categorical

def GetData(fine_tuning=1):
    import tensorflow as tf
    (   (images_train, labels_train),
        (images_test, labels_test),
    ) = tf.keras.datasets.mnist.load_data()
    
    train_size = len(labels_train)
    training_set = int(fine_tuning*train_size)
    labels_train = labels_train[:training_set]
    images_train = images_train[:training_set]
    
    # input_shape = images_train.shape
    labels_test =  to_categorical(labels_test)
    labels_train =  to_categorical(labels_train)

    images_train = images_train.reshape(
        images_train.shape[0], images_train.shape[1], images_train.shape[2], 1
    )
    images_test = images_test.reshape(
        images_test.shape[0], images_test.shape[1], images_test.shape[2], 1
    )
    input_tensor_shape = (images_train.shape[1], images_train.shape[2], 1)
    images_train = images_train.astype("float32")
    images_test = images_test.astype("float32")
    images_train /= 255
    images_test /= 255

    return images_train, images_test, labels_train, labels_test, input_tensor_shape