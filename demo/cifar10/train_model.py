import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from demo.cifar10.Cifar10Task import Cifar10Task
# import tensorflow as tf
# from compression_tools.pruning import model_prune
# from compression_tools.decompose import model_decompose


task = Cifar10Task('demo/cifar10/config.cfg')
model = task.prepare_model('original_model.h5')
compressed_model = task.compress(model)

compressed_model.summary()
# # task.evaluate(model)
# from tensorflow.keras.utils import plot_model
# fig_path = os.path.join(task.config['Model']['model_folder'], 'model.png')
# plot_model(model, to_file=fig_path)

# compressed_model = model_prune(
#                 model,
#                 get_dataset=task.get_dataset,
#                 method="layerwise",
#                 re_method="uniform",
#                 param=0.1,
#                 criterion="magnitude",
#                 min_index=0,
#                 max_index=len(model.layers),
#             )

# compressed_model.summary()
# task.evaluate(compressed_model)
# task.train(model)