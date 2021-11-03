from model.model import CompressModel

model_dir = "/home/wei-bshg/Documents/code/models/"

decompose_setting = {}
decompose_setting["schema"] = "tucker2D"
decompose_setting["rank_selection"] = "VBMF_auto"
decompose_setting["param"] = 0.25
decompose_setting["range"] = [3, -1]
decompose_setting["big_kernel_only"] = True
decompose_setting["option"] = "CL"

my_model = CompressModel(
   original_model=model_dir+'food20_resnet_ft.h5',
   # original_model=model_dir+'food101_res50.h5',
   dataset="food20",
   preprocessing="vgg16")

# my_model.evaluate(my_model.original_model)
my_model.decompose_model(decompose_setting)
my_model.evaluate(my_model.compressed_model)

my_model.fine_tuning(optimizer="default", small_part=0.05)

