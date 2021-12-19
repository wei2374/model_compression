import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
import tensorflow as tf

from SpeechCmdTask import SpeechCmdTask
# from compression_tools.pruning import model_prune


task = SpeechCmdTask('demo/speechcmd/config.cfg')
model = task.prepare_model('trained_model.h5')
model.summary()
# task.evaluate(model)
compressed_model = task.compress(model)
compressed_model.summary()
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