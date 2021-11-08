from model.model import CompressModel

model_dir = "/home/wei-bshg/Documents/code/models/"
import tensorflow as tf
print(tf.test.gpu_device_name())

dataset = "food20"
preprocessing = "vgg16"
pruning_settings = {}
pruning_settings["method"] = "layerwise"
pruning_settings["criterion"] = "gradient1"
pruning_settings["ratio_est"] = "uniform"
pruning_settings["param"] = 0.22
pruning_settings["range"] = [1, -1]
pruning_settings["option"] = "CL"
pruning_settings["big_kernel_only"] = False

my_model = CompressModel(
                # model_dir + "food101_res50.h5",
                # model_dir+"food20_mv2.h5",
                model_dir+"food20_mv2.h5",
                dataset,
                preprocessing)

# my_model.evaluate(my_model.original_model)
# tf.keras.utils.plot_model(my_model.original_model)
my_model.prune_model(pruning_settings)
my_model.evaluate(my_model.compressed_model)
# my_model.run_tflite_inference("ubuntu@192.168.2.127")
my_model.run_tvm_inference("wei@192.168.2.128")
# my_model.fine_tuning(optimizer="default", small_part=1)
