from numpy.lib.arraysetops import isin
from decompose import model_decompose
from pruning import model_prune
# from tools.hardware.tflite_inference import run_tflite
# from tools.hardware.tvm_get_model import RPC_Auto_TVM, open_RPC
# from tools.hardware.tvm_inference import run_tvm
from model_training.training import train_model, validate_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tools.visualization.model_visualization import model_cmp_flops_plot
import tensorflow as tf
import os
import time
from .logger import Logger


class CompressModel:
    def __init__(self,
                original_model,
                dataset,
                preprocessing):
        self.original_model = tf.keras.models.load_model(original_model)
        self.dataset = dataset
        self.preprocessing = preprocessing
        # make folder for logging
        path = os.getcwd()
        self.foldername = os.path.join(path, f'/output/{time.time()}')
        os.makedirs(self.foldername)
        # log the basic information of original model
        self.logger = Logger(self.foldername)
        self.logger.writeln("Information of original model")
        self.evaluate(self.original_model)


    def decompose_model(self, decompose_settings):
        '''
        Call this function to perform model decomposition based on decompose_settings
        range : decompose the layer whose id is within this range, -1 means last layer's id
        schema : decomposition schemes, you can choose from tucker2D, VH, CP, decouple_dp, decouplt_pd,
                channel_output, channel input, channel output, channel_output_nl
        rank_selection : rank selection methods, you can choose from VBMF, BayesOpt, VBMF_auto,
                Param, energy
        param: if the rank_selection methods is not VBMF_auto or BayesOpt, this parameter need to be given
        '''
        if decompose_settings["range"][1] == -1:
            decompose_settings["range"][1] = len(self.original_model.layers)

        self.compressed_model = model_decompose(
                self.original_model,
                self.foldername,
                schema=decompose_settings["schema"],
                rank_selection=decompose_settings["rank_selection"],
                min_index=decompose_settings["range"][0],
                max_index=decompose_settings["range"][1],
                param=decompose_settings["param"],
            )
        model_cmp_flops_plot(self.original_model, self.compressed_model, self.foldername)
        self.logger.log_decomposition(decompose_settings)
        self.evaluate(self.compressed_model)


    def prune_model(self, pruning_settings):
        '''
        Call this function to perform channel pruning based on pruning_settings
        range : decompose the layer whose id is within this range, -1 means last layer's id
        method : pruning methods, you can choose from layerwise_pruning, lasso_pruning and whole_pruning
        ratio_est : prune ratio estimation methods, you can choose uniform, energy, VBMF, BayesOpt
        param : prune ratio estimation methods need this parameter
        '''
        if pruning_settings["range"][1] == -1:
            pruning_settings["range"][1] = len(self.original_model.layers)

        self.compressed_model = model_prune(
                self.original_model,
                dataset=self.dataset,
                method=pruning_settings["method"],
                re_method=pruning_settings["ratio_est"],
                criterion=pruning_settings["criterion"],
                param=pruning_settings["param"],
                min_index=pruning_settings["range"][0],
                max_index=pruning_settings["range"][1],
                foldername=self.foldername
            )
        # model_cmp_flops_plot(self.original_model, self.compressed_model, self.foldername)
        self.logger.log_pruning(pruning_settings)

    def evaluate(self, model):
        self.accuracies, self.flops, self.param = validate_model(
                    model=model,
                    dataset=self.dataset,
                    preprocessing=self.preprocessing)

        self.logger.writeln("Model evaluation")
        self.logger.writeln(f"Model accuracy is {self.accuracies}")
        self.logger.writeln(f"Model FLOPs is {self.flops}")
        self.logger.writeln(f"Model parameter is {self.param}")


    def fine_tuning(
                self,
                optimizer,
                small_part,
                bs=8,
                epoch_n=30,
                ):
        self.logger.writeln("Start fine-tuning")
        history, _, _ = train_model(
            self.compressed_model,
            foldername=self.foldername,
            method=optimizer,
            bs=bs,
            dataset=self.dataset,
            epoch_n=epoch_n,
            small_part=small_part,
            preprocessing=self.preprocessing)
        self.logger.writeln(f"Fine-tuned accuracy is {history.history['val_top1']}")

        print("Fine-tuning is finished")

    def save_model(self):
        foldername = self.foldername
        model = self.compressed_model
        model.save(foldername+"/compressed_model.h5")

    def save_tflite_model(self):
        foldername = self.foldername
        model = self.compressed_model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(foldername+"/tflite_model.tflite", 'wb') as f:
            f.write(tflite_model)
        self.tflite_model = foldername+"/tflite_model.tflite"

    def save_tvm_model(self):
        foldername = self.foldername
        model = self.original_model
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec([1, 224, 224, 3], model.inputs[0].dtype))

        # Get frozen graph def
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        frozen_graph_filename = "graph"
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=foldername,
                          name=f"{frozen_graph_filename}.pb",
                          as_text=False)
        self.tvm_model = foldername+f"/{frozen_graph_filename}.pb"


    def run_tflite_inference(self, server):
        '''
        call this function to measure the inference time and accuracy of compressed model
        with tf lite framework
        '''
        print("Transfroming the model into .tflite format...")
        self.save_tflite_model()
        # run_tflite(self.tflite_model, server)

    def run_tvm_inference(self, server):
        '''
        call this function to measure the inference time and accuracy of compressed model
        with TVM framework
        '''
        print("Transfroming the model into graph format...")
        self.save_tvm_model()
        foldername=self.foldername
        print("Start the auto TVM tuning...")
        # open_RPC(server)
        # RPC_Auto_TVM(
        #     self.tvm_model,
        #     foldername,
        #     hardware="rasp4",
        #     device_key="rk3399")
        # files = [foldername+"/deploy_lib.so", foldername+"/deploy_param.params", foldername+"/deploy_graph.json"]
        # run_tvm(files, server)
        