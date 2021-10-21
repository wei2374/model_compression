import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.eager import profiler
import datetime
import numpy as np
import json
from keras_preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from dataset.food101 import get_food101_data
tf.compat.v1.disable_eager_execution()

img_path = '/home/wei-bshg/Documents/code/tf_with_dec/profilers/images/cat.thumb'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
input = preprocess_input(x)

def timeline_profiler(model, filename, dataset="food20"):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if dataset=="food20":
        train_data, V, _ = get_food101_data(dataset_dir="/home/wei-bshg/Documents/datasets/food-20", batch_size=1, validation_split=0.01)
    
    train_data.reset()
    X_train, y_train = next(train_data)
    for i in range(100): #1st batch is alread fetched before the for loop
        img, label = next(train_data)
        X_train = np.append(X_train, img, axis=0 )
        y_train = np.append(y_train, label, axis=0)
        
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    with tf.device('/cpu:0'):
        model.compile(loss='sparse_categorical_crossentropy',options=run_options, run_metadata=run_metadata)
        output = model.predict_generator(X_train, workers=1,  use_multiprocessing=False, verbose=1)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(filename, 'w') as f:
        f.write(ctf)
    
    print("Timeline profiler finishes its work, result is written into "+filename)

def tensorboard_profiler(model, filename):
    log_dir = filename 
    with tf.device('/cpu:0'):
        profiler.start()
        output = model.predict(input)
        profiler_result = profiler.stop()

    print("Tensorboard profiler finishes its work, result is written into "+filename)


def tensorflow_profiler(model, filename):
    run_metadata = tf.compat.v1.RunMetadata()
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    profile_opts = tf.compat.v1.profiler.ProfileOptionBuilder(tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()).with_file_output(filename).build()
    
    with tf.device('/cpu:0'):
        model.compile(loss='sparse_categorical_crossentropy',options=run_options, run_metadata=run_metadata)
        output = model.predict(input)

    tf.compat.v1.profiler.profile(
        run_meta= run_metadata,  
        cmd='op',
        options=profile_opts
    )
    print("Tensorboard profiler finishes its work, result is written into "+filename)
