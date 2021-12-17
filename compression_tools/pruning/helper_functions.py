import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Add


def get_layer_index(dic, layer_target):
    for index, layer in dic.items():
        if layer == layer_target:
            return index


def load_model_param(model):
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


def find_conv2D_index_before(layer, layer_index_dic):
    if(len(layer.outbound_nodes) == 2):
        return None

    else:
        fore_layer = layer.inbound_nodes[0].inbound_layers
        while(not isinstance(fore_layer, Conv2D)):
            fore_layer = fore_layer.inbound_nodes[0].inbound_layers

        conv2D_index = get_layer_index(layer_index_dic, fore_layer)
        return conv2D_index


def map_act_to_conv(activation_layers, crits_act, layer_index_dic):
    crits = {}
    for act_layer_index in activation_layers:
        fore_layer = activation_layers[act_layer_index].inbound_nodes[0].inbound_layers
        while(not isinstance(fore_layer, Add) and
              not isinstance(fore_layer, Conv2D)) or\
              isinstance(fore_layer, DepthwiseConv2D):
            fore_layer = fore_layer.inbound_nodes[0].inbound_layers

        if isinstance(fore_layer, Conv2D):
            conv2D_index = get_layer_index(layer_index_dic, fore_layer)
            crits[conv2D_index] = crits_act[act_layer_index]

        elif isinstance(fore_layer, Add):
            input_layers = fore_layer.inbound_nodes[0].inbound_layers
            for input_layer in input_layers:
                conv2D_index = find_conv2D_index_before(input_layer, layer_index_dic)
                if conv2D_index is not None:
                    crits[conv2D_index] = crits_act[act_layer_index]
    return crits


def get_block_result(act_layer,  Tensor_data):
    next_layer = act_layer
    while(not isinstance(next_layer, tf.keras.layers.Add)
          and len(next_layer.outbound_nodes) == 1
          and len(next_layer.outbound_nodes) > 0):
        Tensor_data = next_layer(Tensor_data)
        next_layer = next_layer.outbound_nodes[0].layer

    # Act layer is in the last stage
    if len(next_layer.outbound_nodes) == 0:
        output = next_layer(Tensor_data)
        return None, output, False

    else:
        # Act layer is inside a block
        if isinstance(next_layer, tf.keras.layers.Add):
            branch_outputs = []
            branch_outputs.append(Tensor_data)
            mimic_data = Tensor_data
            branch_outputs.append(mimic_data)
            add_layer = next_layer
            Tensor_data = add_layer(branch_outputs)
            next_act_layer = add_layer.outbound_nodes[0].layer
            return next_act_layer, Tensor_data, True

        # Act layer is before a block
        elif len(next_layer.outbound_nodes) > 1:
            Input = next_layer(Tensor_data)
            branch_outputs = []
            for node in next_layer.outbound_nodes:
                next_layer = node.layer
                Tensor_data = Input
                while(not isinstance(next_layer, tf.keras.layers.Add)):
                    Tensor_data = next_layer(Tensor_data)
                    next_layer = next_layer.outbound_nodes[0].layer
                branch_outputs.append(Tensor_data)
                add_layer = next_layer
            Tensor_data = add_layer(branch_outputs)
            next_act_layer = add_layer.outbound_nodes[0].layer
            return next_act_layer, Tensor_data, True


def get_flops_per_channel(model):
    flops = {}
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D):
            C_in = layer.get_weights()[0].shape[2]
            C_out = layer.get_weights()[0].shape[3]
            Kernel = layer.get_weights()[0].shape[0] * layer.get_weights()[0].shape[1]
            H = layer.input_shape[1]
            W = layer.input_shape[2]
            FLOPS_1 = 2*H*W*C_in*C_out*Kernel/C_out

            i = index+1
            FLOPS_2 = 0
            while(i < len(model.layers)):
                if isinstance(model.layers[i], tf.compat.v1.keras.layers.Conv2D):
                    layer2 = model.layers[i]
                    C_in = layer2.get_weights()[0].shape[2]
                    C_out = layer2.get_weights()[0].shape[3]
                    Kernel = layer2.get_weights()[0].shape[1] * layer2.get_weights()[0].shape[0]
                    H = layer2.input_shape[1]
                    W = layer2.input_shape[2]
                    FLOPS_2 = 2*H*W*C_in*C_out*Kernel/C_in
                    break
                else:
                    i += 1

            flops[index] = float(FLOPS_1 + FLOPS_2)/(10 ** 6)
    return flops


def rel_error(A, B):
    """calcualte relative error"""
    return np.mean((A - B) ** 2) ** .5 / np.mean(A ** 2) ** .5
