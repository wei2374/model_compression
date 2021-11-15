import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)

import tensorflow as tf
from pruning import model_prune
from model_training.training import train_model, validate_model
from demo import model_dir, dataset_dir
from keras_flops import get_flops

model_path = os.path.join(model_dir,'original_model.h5')
original_model = tf.keras.models.load_model(model_path)

flops = get_flops(original_model, batch_size=1)
compressed_model = model_prune(
                original_model,
                dataset=dataset_dir,
                method="layerwise",
                re_method="uniform",
                param=0.5,
                criterion="gradient1",
                min_index=1,
                max_index=len(original_model.layers),
            )
new_flops = get_flops(compressed_model, batch_size=1)

# validate_model(dataset=dataset_dir, model=compressed_model)

# fine_tune the model
train_model(raw_model=compressed_model,
            method='sgd',
            small_part=1,
            dataset=dataset_dir,
            foldername=model_dir)