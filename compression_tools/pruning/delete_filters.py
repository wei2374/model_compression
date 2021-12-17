from compression_tools.pruning.helper_functions import get_layer_index
import tensorflow as tf
import numpy as np


def delete_conv2d_output(new_model_param, layer_bias, index, filter, soft_prune=False):
    if not soft_prune:
        new_model_param[index][0] = np.delete(new_model_param[index][0], filter, axis=3)
        if layer_bias:
            new_model_param[index][1] = np.delete(new_model_param[index][1], filter, axis=0)
    else:
        new_model_param[index][0][:, :, :, filter] = float(0)
        if layer_bias:
            new_model_param[index][1][filter] = float(0)


def delete_conv2d_intput(new_model_param, index, filter, soft_prune=False):
    if not soft_prune:
        new_model_param[index][0] = np.delete(new_model_param[index][0], filter, axis=2)
    # else:
    #     initializer = tf.keras.initializers.GlorotNormal(seed=None)
    #     new_model_param[index][0][:, :, filter, :] = initializer(
    #         shape=new_model_param[index][0][:, :, filter, :].shape)


def delete_dense_input(new_model_param, index, filter, soft_prune=False):
    if not soft_prune:
        new_model_param[index][0] = np.delete(new_model_param[index][0], filter, axis=0)
    # else:
    #     initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    #     new_model_param[index][0][filter, :] = initializer(
    #         shape=new_model_param[index][0][filter, :].shape)


def delete_dense_output(new_model_param, index, filter, layer_bias, soft_prune=False):
    if not soft_prune:
        new_model_param[index][0] = np.delete(new_model_param[index][0], filter, axis=1)
        if layer_bias:
            new_model_param[index][1] = np.delete(new_model_param[index][1], filter, axis=0)
    # else:
    #     initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
    #     new_model_param[index][0][:, filter] = initializer(
    #         shape=new_model_param[index][0][:, filter].shape)
    #     if layer_bias:
    #         new_model_param[index][1] = 0


def delete_bn_output(new_model_param, index, filter, soft_prune=False):
    if not soft_prune:
        new_model_param[index][0] = np.delete(new_model_param[index][0], filter)
        new_model_param[index][1] = np.delete(new_model_param[index][1], filter)
        new_model_param[index][2] = np.delete(new_model_param[index][2], filter)
        new_model_param[index][3] = np.delete(new_model_param[index][3], filter) 
    # else:
    #     new_model_param[index][0][filter] = float(1)
    #     new_model_param[index][1][filter] = float(0)
    #     new_model_param[index][2][filter] = float(0)
    #     new_model_param[index][3][filter] = float(1)


def get_down_left_layer(layer_index_dic, layer_index):
    left_layer = layer_index_dic[layer_index].outbound_nodes[0].layer
    return left_layer


def get_down_right_layer(layer_index_dic, layer_index):
    right_layer = layer_index_dic[layer_index].outbound_nodes[1].layer
    return right_layer


def get_up_layers(layer_index_dic, layer_index):
    up_layers = layer_index_dic[layer_index].inbound_nodes[0].inbound_layers
    return up_layers


def get_up_left_layer(layer_index_dic, layer_index):
    left_layer = layer_index_dic[layer_index].inbound_nodes[0].inbound_layers[1]
    return left_layer


def get_up_right_layer(layer_index_dic, layer_index):
    right_layer = layer_index_dic[layer_index].inbound_nodes[0].inbound_layers[0]
    return right_layer


def up_delete_until_conv2D(layer_index_dic, new_model_param, layer_index, filter, soft_prune=False):
    layer = layer_index_dic[layer_index]
    while(not isinstance(layer, tf.compat.v1.keras.layers.Conv2D)
          or isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            left_end_index = get_layer_index(layer_index_dic, layer)
            if(new_model_param[layer_index][0].shape[0] == layer_index_dic[layer_index].output_shape[3]):
                delete_bn_output(new_model_param, left_end_index, filter, soft_prune)

        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            left_end_index = get_layer_index(layer_index_dic, layer)
            delete_conv2d_intput(new_model_param, left_end_index, filter, soft_prune)
            new_model_param[left_end_index][1] = np.delete(new_model_param[left_end_index][1], filter, axis=0)
            
        layer = layer.inbound_nodes[0].inbound_layers

    conv_end_index = get_layer_index(layer_index_dic, layer)
    layer_bias = layer.use_bias
    delete_conv2d_output(new_model_param, layer_bias, conv_end_index, filter, soft_prune)


def delete_filter_after(
        new_model_param,
        layer_index,
        layer_index_dic,
        filter,
        soft_prune=False,
        layer_type="conv2D"):
    ''' Delete channels params related with pruned channels

    Args:
        new_model_param: (float list) weights or parameters in each layer
        layer_index: (int) Layer to be pruned
        layer_index_dic: (dictionary) [layer_index:layer]
        filter: pruned channels index
        soft_prune: whether or not use soft_prune strategy

    '''
    current_layer = layer_index_dic[layer_index]
    layer_bias = current_layer.use_bias
    if layer_type == "Dense":
        following_layer = current_layer.outbound_nodes[0].layer
        if isinstance(following_layer, tf.keras.layers.Dense):
            following_index = get_layer_index(layer_index_dic, following_layer)
            delete_dense_output(new_model_param, layer_index, filter, layer_bias, soft_prune)
            delete_dense_input(new_model_param, following_index, filter, soft_prune)
            return new_model_param
        else:
            return new_model_param

    elif layer_type == "conv2D":
        while(not isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.Flatten) and\
            not isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.Dense) and\
            not isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.AveragePooling2D)):
            
            following_layer = current_layer.outbound_nodes[0].layer
            following_index = get_layer_index(layer_index_dic, following_layer)
            if isinstance(following_layer, tf.keras.layers.BatchNormalization):
                if(new_model_param[following_index][0].shape[0] == following_layer.output_shape[3]):
                    delete_bn_output(new_model_param, following_index, filter, soft_prune)

            # The conv2D layer is before a branch
            if(len(following_layer.outbound_nodes) == 2 and
               len(following_layer.inbound_nodes[0].flat_input_ids) == 1):
                branch_layer_index = following_index
                # Delete the output channels of current Conv2D
                delete_conv2d_output(new_model_param, layer_bias, layer_index, filter, soft_prune)

                # Delete the input channels of following left Conv2D
                left_down_conv_layer = get_down_left_layer(layer_index_dic, branch_layer_index)
                left_down_conv_layer_index = get_layer_index(layer_index_dic, left_down_conv_layer)
                if isinstance(left_down_conv_layer, tf.keras.layers.Conv2D):
                    delete_conv2d_intput(
                        new_model_param, left_down_conv_layer_index, filter, soft_prune)

                # Delete the input channels of following right Conv2D
                # elif isinstance(left_down_conv_layer, tf.keras.layers.GlobalAveragePooling2D):
                #     dense_layer_index = get_layer_index(
                #         layer_index_dic, left_down_conv_layer.outbound_nodes[0].layer)
                #     delete_dense_input(new_model_param, dense_layer_index, filter, soft_prune)

                # Delete the input channels of following right Conv2D
                right_down_conv_layer = get_down_right_layer(layer_index_dic, branch_layer_index)
                right_down_conv_layer_index = get_layer_index(
                        layer_index_dic, right_down_conv_layer)
                if isinstance(right_down_conv_layer, tf.keras.layers.Conv2D):
                    delete_conv2d_intput(
                        new_model_param, right_down_conv_layer_index, filter, soft_prune)
                
                # elif isinstance(right_down_conv_layer, tf.keras.layers.GlobalMaxPooling2D):
                #     dense_layer_index = get_layer_index(
                #         layer_index_dic, right_down_conv_layer.outbound_nodes[0].layer)
                #     delete_dense_input(new_model_param, dense_layer_index, filter, soft_prune)
                
                # No Conv layer, direct connect to add layer 
                add_layer_index = get_layer_index(layer_index_dic, right_down_conv_layer)
                if(not isinstance(left_down_conv_layer, tf.keras.layers.Conv2D)):
                    add_layer_index = get_layer_index(layer_index_dic, left_down_conv_layer)
                    
                else:
                    ct_flag1 = True
                    while(ct_flag1):
                        # delete output channels of last conv2D layer on the other branch
                        if(add_layer_index==right_down_conv_layer_index):
                            left_end_index = get_layer_index(
                                layer_index_dic,
                                right_down_conv_layer.inbound_nodes[0].inbound_layers[1])
                            up_delete_until_conv2D(
                                layer_index_dic, new_model_param, left_end_index, filter, soft_prune)
                        else:
                            right_end_index = get_layer_index(
                                layer_index_dic,
                                left_down_conv_layer.inbound_nodes[0].inbound_layers[1])
                            up_delete_until_conv2D(
                                layer_index_dic, new_model_param, right_end_index, filter, soft_prune)
                            
                        act_layer_id = add_layer_index+1
                        
                        left_down_conv_layer = get_down_left_layer(
                                layer_index_dic, act_layer_id)
                        right_down_conv_layer = get_down_right_layer(
                                layer_index_dic, act_layer_id)
                        
                        next_add_layer = left_down_conv_layer
                        if(isinstance(left_down_conv_layer, tf.keras.layers.Conv2D) and\
                            isinstance(right_down_conv_layer, tf.keras.layers.Conv2D)):
                            ct_flag1 = False
                        if(isinstance(left_down_conv_layer, tf.keras.layers.Conv2D)):
                            left_down_conv_layer_index = get_layer_index(
                                layer_index_dic, left_down_conv_layer)
                            delete_conv2d_intput(
                                new_model_param, left_down_conv_layer_index, filter, soft_prune)
                        else:
                            next_add_layer = left_down_conv_layer
                        
                        if(isinstance(right_down_conv_layer, tf.keras.layers.Conv2D)):
                            right_down_conv_layer_index = get_layer_index(
                                layer_index_dic, right_down_conv_layer)
                            delete_conv2d_intput(
                                new_model_param, right_down_conv_layer_index, filter, soft_prune)
                        else:
                            next_add_layer = left_down_conv_layer
                        right_down_conv_layer = next_add_layer

                return new_model_param

            # This conv2D layer is followed by a conv2D
            elif isinstance(following_layer, tf.compat.v1.keras.layers.Conv2D):
                # # print("This Conv2D is before a Conv2D")
                # Delete the input channels of following Conv2D
                delete_conv2d_intput(new_model_param, following_index, filter, soft_prune)
                if (not isinstance(following_layer, tf.keras.layers.DepthwiseConv2D)):
                    # Delete the output channels of current Conv2D
                    delete_conv2d_output(
                        new_model_param, layer_bias, layer_index, filter, soft_prune)
                    return new_model_param
                else:
                    new_model_param[following_index][1] = np.delete(new_model_param[following_index][1], filter, axis=0)
                    # return new_model_param

            # This conv2D layer is followed by an Add layer
            elif isinstance(following_layer, tf.keras.layers.Add):
                add_layer = following_layer

                up_layer = get_up_layers(layer_index_dic, layer_index)
                # right layer
                if len(up_layer.outbound_nodes) == 2:
                    # # print("This is a conv2D at right up position of a add layer")
                    # 1. Delete the output channels of current Conv2D
                    delete_conv2d_output(
                        new_model_param, layer_bias, layer_index, filter, soft_prune)
                    final_flag = False
                    if True:
                        next_layer = following_layer.outbound_nodes[0].layer
                        while(len(next_layer.outbound_nodes) != 2):
                            next_layer = next_layer.outbound_nodes[0].layer
                            if(isinstance(next_layer, tf.keras.layers.AveragePooling2D)):
                                end_layer = next_layer
                                layer_index = get_layer_index(layer_index_dic, end_layer)

                                while(not isinstance(end_layer, tf.keras.layers.Dense)):
                                    end_layer = end_layer.outbound_nodes[0].layer
                                dense_layer = end_layer
                                dense_layer_index = get_layer_index(layer_index_dic, dense_layer)
                                layer_output = layer_index_dic[dense_layer_index-2]
                                layer_output_shape = layer_output.output_shape
                                shape = (layer_output_shape[1]*layer_output_shape[2])

                                filters = []
                                channels = layer_output_shape[3]
                                new_filter = filter[0]
                                for s in range(shape):
                                    filters = np.concatenate([filters, new_filter])
                                    new_filter = new_filter+channels
                                filters = [int(i) for i in filters]
                                delete_dense_input(new_model_param, dense_layer_index, filters, soft_prune)
                                final_flag = True
                                break
                                # return new_model_param

                        # Start of next block
                        end_layer = next_layer
                        continue_flag = True
                        while(continue_flag):
                            next_layer_index = get_layer_index(layer_index_dic, end_layer)
                            if final_flag:
                                continue_flag = False
                                before_layer = end_layer
                            else:
                                before_layer = end_layer.inbound_nodes[0].inbound_layers

                            if len(end_layer.outbound_nodes) == 2:
                                # 1. Delete the input channels of following LEFT Conv2D
                                left_down_conv_layer = get_down_left_layer(
                                    layer_index_dic, next_layer_index)
                                if isinstance(
                                        left_down_conv_layer, tf.compat.v1.keras.layers.Conv2D):
                                    left_down_conv_layer_index = get_layer_index(
                                        layer_index_dic, left_down_conv_layer)
                                    delete_conv2d_intput(
                                            new_model_param, left_down_conv_layer_index,
                                            filter, soft_prune)

                                # 2. RIGHT Conv2D
                                right_down_conv_layer = get_down_right_layer(
                                                    layer_index_dic, next_layer_index)
                                if isinstance(
                                        right_down_conv_layer, tf.compat.v1.keras.layers.Conv2D):
                                    continue_flag = False
                                    right_down_conv_layer_index = get_layer_index(
                                                            layer_index_dic, right_down_conv_layer)
                                    delete_conv2d_intput(
                                        new_model_param, right_down_conv_layer_index,
                                        filter, soft_prune)
                                if(isinstance(
                                        right_down_conv_layer, tf.compat.v1.keras.layers.Conv2D) and\
                                    isinstance(
                                        left_down_conv_layer, tf.compat.v1.keras.layers.Conv2D)):
                                    continue_flag = False
                                    final_flag = True

                            while(not isinstance(before_layer, tf.keras.layers.Add)):
                                before_layer = before_layer.inbound_nodes[0].inbound_layers
                            # 3. left end output
                            left_end_index = get_layer_index(
                                layer_index_dic, before_layer.inbound_nodes[0].inbound_layers[1])
                            up_delete_until_conv2D(
                                layer_index_dic, new_model_param,
                                left_end_index, filter, soft_prune)

                            # find the next block
                            if not final_flag:
                                end_layer = end_layer.outbound_nodes[1].layer
                                while(len(end_layer.outbound_nodes) != 2):
                                    if(isinstance(end_layer, tf.keras.layers.Dense)):
                                        dense_layer_index = get_layer_index(
                                            layer_index_dic, end_layer)
                                        delete_dense_input(
                                            new_model_param, dense_layer_index, filter, soft_prune)
                                        final_flag = True
                                        fore_layer = end_layer.inbound_nodes[0].inbound_layers
                                        while(not isinstance(fore_layer, tf.keras.layers.Add)):
                                            fore_layer = fore_layer.inbound_nodes[0].inbound_layers
                                        end_layer = fore_layer
                                        break
                                    end_layer = end_layer.outbound_nodes[0].layer
                    return new_model_param

                else:
                    # # print("LAST LEFT")
                    # add_layer = following_layer
                    # next_layer = add_layer.outbound_nodes[0].layer
                    # while(len(next_layer.outbound_nodes) == 1):
                    #     if isinstance(next_layer, tf.keras.layers.AveragePooling2D):
                    #         dense_layer_index = get_layer_index(layer_index_dic, next_layer)
                    #         # # print("Got a Dense Layer at the end")
                    #         break

                    #     next_layer = next_layer.outbound_nodes[0].layer
                    return new_model_param
            current_layer = following_layer

        # # print("Got a Dense Layer at the end")
        # Delete the output channels of current Conv2D
        if isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.AveragePooling2D):
            delete_conv2d_output(new_model_param, layer_bias, layer_index, filter, soft_prune)
            while(not isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.Dense)):
                current_layer = current_layer.outbound_nodes[0].layer
            dense_layer = current_layer.outbound_nodes[0].layer
            dense_layer_index = get_layer_index(layer_index_dic, dense_layer)
            layer_output = layer_index_dic[dense_layer_index-2]
            layer_output_shape = layer_output.output_shape
            shape = (layer_output_shape[1]*layer_output_shape[2])

            filters = []
            channels = layer_output_shape[3]
            new_filter = filter[0]
            for s in range(shape):
                filters = np.concatenate([filters, new_filter])
                new_filter = new_filter+channels
            filters = [int(i) for i in filters]
            delete_dense_input(new_model_param, dense_layer_index, filters, soft_prune)

        # if isinstance(current_layer.outbound_nodes[0].layer, tf.keras.layers.Dense):
        #     delete_conv2d_output(new_model_param, layer_bias, layer_index, filter, soft_prune)
        #     dense_layer = current_layer.outbound_nodes[0].layer
        #     dense_layer_index = get_layer_index(layer_index_dic, dense_layer)
        #     layer_output = layer_index_dic[dense_layer_index-2]
        #     layer_output_shape = layer_output.output_shape
        #     shape = (layer_output_shape[1]*layer_output_shape[2])

        #     filters = []
        #     channels = layer_output_shape[3]
        #     new_filter = filter[0]
        #     for s in range(shape):
        #         filters = np.concatenate([filters, new_filter])
        #         new_filter = new_filter+channels
        #     filters = [int(i) for i in filters]
        #     delete_dense_input(new_model_param, dense_layer_index, filters, soft_prune)
        return new_model_param


def delete_filter_before(
                    new_model_param,
                    layer_types,
                    layer_output_shape,
                    layer_bias,
                    layer_index,
                    filter,
                    layer_index_dic):
    if layer_types[layer_index] == "Conv2D":
        current_layer = layer_index_dic[layer_index]
        fore_layer = current_layer.inbound_nodes[0].inbound_layers
        while((not isinstance(fore_layer, tf.keras.layers.Conv2D)
              or isinstance(fore_layer, tf.keras.layers.DepthwiseConv2D)
              and not isinstance(fore_layer, tf.keras.layers.Add))
              and not len(fore_layer.outbound_nodes) == 2):
            fore_layer = fore_layer.inbound_nodes[0].inbound_layers

        # inside a a sequence
        if isinstance(fore_layer, tf.compat.v1.keras.layers.Conv2D):
            # # print("This conv2D is inside a sequence")
            fore_layer = current_layer.inbound_nodes[0].inbound_layers
            fore_layer_index = get_layer_index(layer_index_dic, fore_layer)
            up_delete_until_conv2D(
                layer_index_dic, new_model_param, fore_layer_index, filter)
            return new_model_param, layer_output_shape

        elif isinstance(fore_layer, tf.keras.layers.Add):
            # print("This conv2D is before an add layer, ignore")
            return new_model_param, layer_output_shape

        elif len(fore_layer.outbound_nodes) == 2:
            # print("This conv2D is at the beginning edge")

            next_layer = current_layer.outbound_nodes[0].layer
            while(not isinstance(next_layer, tf.compat.v1.keras.layers.Conv2D)
                  and not isinstance(next_layer, tf.keras.layers.Add)):
                next_layer = next_layer.outbound_nodes[0].layer

            if isinstance(next_layer, tf.compat.v1.keras.layers.Conv2D):
                # print("left edge")
                pass

            elif isinstance(next_layer, tf.keras.layers.Add):
                # print("right edge")
                branch_layer = current_layer.inbound_nodes[0].inbound_layers
                continue_flag = True
                while(continue_flag):
                    # print("a block before")
                    # Delete neighbor
                    branch_layer_index = get_layer_index(layer_index_dic, branch_layer)
                    left_conv_layer = get_down_left_layer(layer_index_dic, branch_layer_index)
                    left_conv_layer_index = get_layer_index(layer_index_dic, left_conv_layer)
                    delete_conv2d_intput(new_model_param, left_conv_layer_index, filter)

                    fore_layer = branch_layer.inbound_nodes[0].inbound_layers
                    while(not isinstance(fore_layer, tf.keras.layers.Add)
                          and not isinstance(fore_layer, tf.keras.layers.Conv2D)):
                        fore_layer = fore_layer.inbound_nodes[0].inbound_layers

                    if isinstance(fore_layer, tf.keras.layers.Conv2D):
                        first_base_index = get_layer_index(layer_index_dic, branch_layer)
                        up_delete_until_conv2D(
                            layer_index_dic, new_model_param, first_base_index, filter)
                        break

                    add_layer_index = get_layer_index(layer_index_dic, fore_layer)

                    # left side
                    left_end_layer = get_up_left_layer(layer_index_dic, add_layer_index)
                    left_end_layer_index = get_layer_index(layer_index_dic, left_end_layer)
                    up_delete_until_conv2D(
                        layer_index_dic, new_model_param, left_end_layer_index, filter)

                    # right side
                    right_end_layer = get_up_right_layer(layer_index_dic, add_layer_index)
                    right_end_layer_index = get_layer_index(layer_index_dic, right_end_layer)
                    if(len(right_end_layer.outbound_nodes) == 1):
                        # print("End point, break")
                        continue_flag = False
                        up_delete_until_conv2D(
                            layer_index_dic, new_model_param,
                            right_end_layer_index, filter)
                    else:
                        # print("continue")
                        branch_layer = right_end_layer

    else:
        # print("No conv layer")
        pass

    return new_model_param, layer_output_shape


def get_layer_shape_conv(new_model_param, layer):
    """
    Gets the struture of the new generated model and return the shape of
    the current layer

    Args:
        new_model_param: The params of the new generated model
        layer: the current layer we want the shape from

    Return:
        shape of the current layer
    """
    return new_model_param[layer][0].shape[2]
