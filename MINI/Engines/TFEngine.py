from MINI.Core.Engine import Engine
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, DepthwiseConv2D,\
    BatchNormalization, Activation, ZeroPadding2D, Add
import numpy as np
from MINI.Tools.utils.utils import get_block_result


class TFEngine(Engine):
    def get_weights(self, layer=None, model=None, index=None):
        if layer is not None:
            return np.asarray(layer.get_weights())
        return np.asarray(model.layers[index].get_weights())


    def get_prev_layers(self, layer):
        layers = []
        for node in layer.inbound_nodes:
            if(isinstance(node.inbound_layers, list)):
                for _layer in node.inbound_layers:
                    layers.append(_layer)
            else:
                layers.append(node.inbound_layers)
        return layers
     

    def get_next_layers(self, layer):
        layers = []
        for node in layer.outbound_nodes:
            if(isinstance(node.outbound_layer, list)):
                for _layer in node.outbound_layer:
                    layers.append(_layer)
            else:
                layers.append(node.outbound_layer)
        return layers


    def is_branch(self, layer):
        return len(self.get_next_layers(layer.raw_layer))>1

    def is_merge(self, layer):
        return len(self.get_prev_layers(layer.raw_layer))>1

    def is_type(self, layer, type):
        from MINI.Core.Layer import Layer
        if(isinstance(layer, Layer)):
            return isinstance(layer.raw_layer, type)
        else:
            return isinstance(layer, eval(type))

    def wrap(self, layer):
        from MINI.Layers.BNLayer import BNLayer
        from MINI.Layers.Conv2DLayer import Conv2DLayer
        from MINI.Layers.DepthwiseLayer import DepthwiseLayer
        from MINI.Layers.DenseLayer import DenseLayer
        from MINI.Layers.FlattenLayer import FlattenLayer
        from MINI.Core.Layer import Layer
        
        LAYER_MAPPING={
                    BatchNormalization: BNLayer,
                    Conv2D: Conv2DLayer,
                    DepthwiseConv2D: DepthwiseLayer,
                    Dense: DenseLayer,
                    Flatten: FlattenLayer
                }

        if(any([isinstance(layer, layer_type) for layer_type in LAYER_MAPPING])):
            for layer_type in LAYER_MAPPING:
                if(isinstance(layer, layer_type)):
                    layer_cons = LAYER_MAPPING.get(layer_type)
                    return layer_cons
        else:
            return Layer
    
    def load_model_param(self, model):
        layer_params = []
        layer_types = []
        layer_output_shape = []
        layer_bias = []
        layer_index_dic = {}

        for index, layer in enumerate(model.layers):
            layer_types.append(layer.__class__.__name__)
            layer_params.append(layer.get_weights())
            layer_index_dic[index] = layer
            layer_output_shape.append(list(layer.output_shape))
            try:
                layer_bias.append(layer.use_bias)
            except Exception:
                layer_bias.append(None)
        return np.array(layer_types), np.array(layer_params),\
            layer_output_shape, layer_bias, layer_index_dic

    def use_bias(self, layer):
        return layer.raw_layer.use_bias
    
    def execute(self, layer, data):
        return layer.raw_layer(data)

    def get_intermediate_output(self, model, layers, train_data):
        if isinstance(train_data, tf.data.Dataset):
            batch = next(iter(train_data))
        else:
            batch = next(train_data)

        intermediate_layer_model = tf.keras.Model(
                inputs=model.input,
                outputs=[layers[layer].output for layer in layers])
        activation_outputs = (intermediate_layer_model(batch[0]))
        return activation_outputs


    def calculate_sec_gradient(self, model, layers, train_data):
        if isinstance(train_data, tf.data.Dataset):
            batch = next(iter(train_data))
        else:
             batch = next(train_data)
        batch = next(train_data)
        if batch[1].ndim==1:
            loss_fun = "tf.metrics.sparse_categorical_crossentropy(batch[1], pred)"
        else:
            loss_fun = "tf.metrics.categorical_crossentropy(batch[1], pred)"
        
        # Get gradients
        grad = {}
        for conv_index in layers:
            Input = tf.convert_to_tensor(batch[0])
            with tf.GradientTape(persistent=True) as t:
                t.watch(Input)
                with tf.GradientTape(persistent=True) as tt:
                    ##########################################
                    # run the following layers
                    tt.watch(Input)
                    Tensor_data = Input
                    pred = model(Tensor_data)
                    ###########################################
                    loss = eval(loss_fun)
                grad1 = tt.gradient(loss, model.layers[conv_index].weights[0])
            grad[conv_index] = t.gradient(grad1, model.layers[conv_index].weights[0])
        
        return grad
    

    def calculate_taylor_gradient(self, model, layers, train_data):
        if isinstance(train_data, tf.data.Dataset):
            batch = next(iter(train_data))
        else:
             batch = next(train_data)
        batch = next(train_data)
        if batch[1].ndim==1:
            loss_fun = "tf.metrics.sparse_categorical_crossentropy(batch[1], pred)"
        else:
            loss_fun = "tf.metrics.categorical_crossentropy(batch[1], pred)"
        
        intermediate_layer_model = tf.keras.Model(
            inputs=model.input,
            outputs=[layers[layer].output for layer in layers])
        
        activation_outputs = intermediate_layer_model(batch[0])
        activation_outputs_dic = {}
        for counter, layer_index in enumerate(layers):
            activation_outputs_dic[layer_index] = activation_outputs[counter]

        # Get taylor gradients
        grad = {}
        for act_index in layers:
            Input = tf.convert_to_tensor(activation_outputs_dic[act_index])
            with tf.GradientTape(persistent=True) as t:
                t.watch(Input)
                Tensor_data = Input
                act_layer = layers[act_index]

                continue_flag = True
                while(continue_flag):
                    act_layer, Tensor_data, continue_flag = get_block_result(
                                                act_layer, Tensor_data)
                
                pred = Tensor_data
                loss = eval(loss_fun)
            grad[act_index] = t.gradient(loss, Input)

        return grad, activation_outputs_dic