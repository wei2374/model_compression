import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from Food101Task import Food101Task
import tensorflow as tf
from compression_tools.pruning import model_prune


task = Food101Task('demo/food20/config.cfg')
model = task.prepare_model(filename='trained_model.h5')
model.summary()
# task.evaluate(model)

compressed_model = model_prune(
                model,
                get_dataset=task.get_dataset,
                method="layerwise",
                re_method="uniform",
                param=0.6,
                criterion="magnitude",
                min_index=0,
                max_index=len(model.layers),
            )
task.evaluate(model)
task.train(model)