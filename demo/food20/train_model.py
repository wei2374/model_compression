import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from Food101Task import Food101Task


task = Food101Task('demo/food20/config.cfg')
model = task.prepare_model(type="mobilenet_v2")#filename='trained_model.h5')
model.summary()

compressed_model = task.compress(model)
compressed_model.summary()
# compressed_model = model_prune(
#                 model,
#                 get_dataset=task.get_dataset,
#                 method="layerwise",
#                 re_method="uniform",
#                 param=0.6,
#                 criterion="magnitude",
#                 min_index=0,
#                 max_index=len(model.layers),
#             )
# task.evaluate(model)
# task.train(model)