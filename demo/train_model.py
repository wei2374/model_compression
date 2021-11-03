import autokeras as ak
from model.resnet import ResNet
from model.mobilenet import MobileNet
from model_training.training import train_model, validate_model

model_dir = "/home/wei-bshg/Documents/code/models/"
# model = ResNet(
#     input_shape=(224, 224, 3), num_classes=101, models_filename= model_dir+'resnet50_best_model.hdf5',)
model = MobileNet(
    input_shape=(224, 224, 3), num_classes=101)

model.summary()
model.save(model_dir+"food101_mv1.h5")
history, _, _ = train_model(
            model,
            foldername="/home/wei-bshg/Documents/code/models",
            method="cycling",
            bs=16,
            dataset="food101",
            epoch_n=30,
            small_part=1,
            preprocessing="vgg16")