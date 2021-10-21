from model.model import CompressModel


model_dir = "/home/wei-bshg/Documents/code/models/"
# my_model = tf.keras.models.load_model(model_dir+"food20_resnet_ft.h5", compile=False)
# my_model = tf.keras.models.load_model(model_dir+"food20_mv2.h5", compile=False)

dataset = "food20"
preprocessing = "vgg16"
pruning_settings = {}
pruning_settings["method"] = "lasso"
pruning_settings["criterion"] = "activation"
pruning_settings["ratio_est"] = "uniform"
pruning_settings["param"] = 0.22
pruning_settings["range"] = [1, -1]
pruning_settings["option"] = "CL"
pruning_settings["big_kernel_only"] = False

my_model = CompressModel(
                    model_dir+"food20_resnet_ft.h5",
                    dataset,
                    preprocessing)

my_model.prune_model(pruning_settings)
my_model.evaluate(my_model.compressed_model)
my_model.fine_tuning(optimizer="default", small_part=0.05)
