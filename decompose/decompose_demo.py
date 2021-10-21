import tensorflow as tf
# from tools.net_flops import net_flops
from model.model import CompressModel

model_dir = "/home/wei-bshg/Documents/code/models/"
# my_model = tf.keras.models.load_model(model_dir+"food20_resnet_ft.h5", compile=False)
my_model = tf.keras.models.load_model(model_dir+'food20_VGG16_ft.h5', compile=False)


decompose_setting = {}
decompose_setting["schema"] = "channel_all"
decompose_setting["rank_selection"] = "Param"
decompose_setting["param"] = 0.25
decompose_setting["range"] = [3, -1]
decompose_setting["big_kernel_only"] = True
decompose_setting["option"] = "CL"

my_model = CompressModel(
   original_model=model_dir+'food20_VGG16_ft.h5',
   dataset="food20",
   preprocessing="vgg16")

my_model.decompose_model(decompose_setting)
# my_model.evaluate(my_model.compressed_model)
my_model.fine_tuning(optimizer="cycling", small_part=0.05)
# if rank_selection == "PCA":
#     ranks = estimate_ranks_PCA(my_model)
# if rank_selection == "measurement":
#     ranks = estimate_ranks_measurement(my_model, param)
# else:
#     ranks = None

