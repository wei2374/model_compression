import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)

import tensorflow as tf
from decompose import model_decompose
from model_training.training import train_model, validate_model
from demo import model_dir, dataset_dir
from tools.visualization.model_visualization import visualize_model
model_path = os.path.join(model_dir,'original_model.h5')
original_model = tf.keras.models.load_model(model_path)

validate_model(dataset=dataset_dir, model=original_model)

compressed_model = model_decompose(
                original_model,
                schema="depthwise_dp",
                rank_selection="energy",
                param=0.9,
                min_index=3,
                max_index=len(original_model.layers),
            )

validate_model(dataset=dataset_dir, model=compressed_model)

# fine_tune the model
train_model(raw_model=compressed_model,
            method='sgd',
            dataset=dataset_dir,
            foldername=model_dir)