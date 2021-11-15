import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from model.resnet import ResNet
from model.vgg16 import VGG16
from model_training.training import train_model, validate_model
from demo import model_dir, dataset_dir

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model = ResNet(
    input_shape=(224, 224, 3), num_classes=20)

# model = VGG16(input_shape=(224, 224, 3), num_classes=20)
history, _, _ = train_model(
            model,
            model_name='original_model.h5',
            foldername=model_dir,
            method="sgd",
            bs=32,
            dataset=dataset_dir,
            epoch_n=30,
            small_part=1,
            preprocessing="vgg16")
