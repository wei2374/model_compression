import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import preprocess_input

def performance_profiler(my_model, dataset, metrics_array, validation_prop):
    val_idg = ImageDataGenerator(preprocessing_function=preprocess_input, \
        validation_split=validation_prop)

    val_gen1 = val_idg.flow_from_directory(
        dataset+"/val",
        target_size=(224, 224),
        batch_size=32,
        subset='validation'
        )

    my_model.compile(loss='categorical_crossentropy', metrics=[metrics_array])
    with tf.device('/cpu:0'):
        result = my_model.evaluate(val_gen1)
    
    return dict(zip(my_model.metrics_names, result))

def get_inference_time(raw_model, dataset):
    if dataset == "food20":
        _, valid_data = get_data_from_dataset(dataset, validation_split=0.01)

    with tf.device('/cpu:0'):
        raw_model.predict_generator(
            valid_data,
            workers=1,
            use_multiprocessing=False,
            verbose=1)