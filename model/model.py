from decompose.model_modifier import model_decompose
from pruning.model_modifier import model_prune
from model_training.training import train_model, validate_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import os
import time
from .logger import Logger


class CompressModel:
    def __init__(
                self,
                original_model,
                dataset,
                preprocessing):
        self.original_model = tf.keras.models.load_model(original_model)
        self.dataset = dataset
        self.preprocessing = preprocessing

        path = os.getcwd()
        self.foldername = path+f'/output/{time.time()}'
        os.makedirs(self.foldername)
        self.logger = Logger(self.foldername)
        self.logger.writeln("Information of original model")
        self.logger.log_model(self.original_model)

    def decompose_model(self, decompose_settings):
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
                big_kernel_only=decompose_settings["big_kernel_only"],
            )
        self.logger.log_decomposition(decompose_settings)

    def prune_model(self, pruning_settings):
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
                big_kernel_only=pruning_settings["big_kernel_only"],
                option=pruning_settings["option"],
                foldername=self.foldername
            )
        self.logger.log_pruning(pruning_settings)

    def evaluate(self, model):
        self.accuracies, self.flops, self.param = validate_model(
                    model=model,
                    dataset=self.dataset,
                    preprocessing=self.preprocessing)
        self.logger.writeln(f"Evaluation accuracy is {self.accuracies}")

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
        self.logger.writeln(f"Fine-tuned accuracy is ")

        print("Fine-tuning is finished")

    def save_model(self):
        foldername = self.foldername
        model = self.compressed_model
        # for PC inference
        model.save(foldername+"/compressed_model.h5")

        # for tflite inference
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(foldername+"/tflite_model.tflite", 'wb') as f:
            f.write(tflite_model)

        # for tvm inference
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec([1, 224, 224, 3], model.inputs[0].dtype))

        # Get frozen graph def
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
        frozen_graph_filename = "graph"
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=foldername,
                          name=f"{frozen_graph_filename}.pb",
                          as_text=False)
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=foldername,
                          name=f"{frozen_graph_filename}.pbtxt",
                          as_text=True)
