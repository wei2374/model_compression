# model_compression
This repo contains implementation of several tensor decomposition and channel pruning algorithms

# Requirement
NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2

# Usage
## Dataset
* First we need to setup the dataset in a certain format, we need to set the folder's name in **demo/config.cfg** file
* The dataset(food20 for example) inside the folder would look like    

  dataset

            ----->  images/
                            -----> class1/
                            ------> class2/
                            ...
            ----->  meta/\
                            ------> train.txt
                            ------> test.txt
* Then we partition the the dataset into train and test by running **demo/prepare_data.py**, it becomes
    dataset
    
            ----->  images/
                            ------> class1/
                            ------> class2/
                            ...
            ----->  meta/
                            ------> train.txt
                            ------> test.txt
            ----->  train/
                            ------> class1/
                            ------> class2/
                            ...
            ----->  test/
                            ------> class1/
                            ------> class2/
                            ...

## Train an original model
* Define the model to be trained in folder **model**
* Train the model by running **demo/train_model.py**

## Compress the model
* To compress the model, we can either decompose it or prune the channel, there are two way of doing it

### 1. Using model_prune/model_decompose function
* This method directly call the function of model compress, it is more lightweight. The implementation is found in **demo/compress_function_demo/decompose_demo.py** and **demo/compress_function_demo/pruning_demo.py**
* The fine_tuned model will be found in *model_dir* (define in config.cfg)

## Using CompressModel class
* The example using compress model class can be found in the demo folder
* This method creats a folder for each model compression action, the log and graph will be saved in the folder


