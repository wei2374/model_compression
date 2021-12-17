import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from VwwTask import VwwTask
import tensorflow as tf
from compression_tools.pruning import model_prune


task = VwwTask('demo/vww/config.cfg')
model = task.prepare_model('original_model.h5')

model.summary()
# task.evaluate(model)

compressed_model = model_prune(
                model,
                get_dataset=task.get_dataset,
                method="layerwise",
                re_method="uniform",
                param=0.6,
                criterion="random",
                min_index=0,
                max_index=len(model.layers),
            )
compressed_model.summary()
task.evaluate(compressed_model)

task.train(model)