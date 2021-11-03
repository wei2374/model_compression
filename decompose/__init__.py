import tensorflow as tf
from tensorflow.keras.models import Model
from decompose.decompose_schemes.decomposition_methods import decompose_conv2d_layer
from decompose.rank_estimation.rank_estimator import estimate_rank
from tools.visualization.decompose_visualization_tool import plot_rank_percentage
from tools.progress.bar import Bar
from tools.visualization.model_visualization import plot_model_decompose


def decompose_layers(
        original_model,
        foldername,
        schema="tucker2D",
        min_index=0,
        max_index=0,
        rank_selection="VBMF",
        param=0.2,
        option="CL",
        big_kernel_only=False
        ):
    '''
    decompose layers based on decomosition settings

    Key arguments:
    original_model -- model to be decomposed
    foldername -- folder used to save related logging and pictures
    schema -- decomposition schemes
    rank_selection -- Methods for rank estimation
    param -- parameter necessary for rank estimation methods
    min_index -- only decompose the layers whose id is behind min_index
    max_index -- only decompose the layers whose id is before min_index
    option -- "CL"-> only decompose convolutional layers
              "FL"-> only decompose fully connected layers
              "CLFL"-> decompose convolutional layers as well as
                     fully connected layers
    big_kernel_only -- if only decompose layer with large receptive field
    Return:
    new_layers -- decomposed layers in a dictionary format
    '''

    kernel_limit = 1 if big_kernel_only else 0
    new_layers = {}
    layer_rank = {}
    layer_ranks_all = {}
    target_layers = {}

    # get target layers
    for index, layer in enumerate(original_model.layers):
        if isinstance(layer, tf.compat.v1.keras.layers.Conv2D) \
                and layer.kernel_size[0] > kernel_limit\
                and index >= min_index\
                and index <= max_index\
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)\
                and (option == "CL" or option == "CLFL"):
            target_layers[index] = layer
    layer_num = len(target_layers.items())

    # estimate ranks
    with Bar(f'Rank estimation with {rank_selection} ...') as bar:
        for index in target_layers:
            rank = estimate_rank(target_layers[index], rank_selection, param, schema)
            input_channels = target_layers[index].input_shape[3]
            output_channels = target_layers[index].output_shape[3]
            layer_rank[index] = rank
            layer_ranks_all[index] = [input_channels, output_channels, rank[0], rank[1]]
            bar.next((100/layer_num))
    plot_rank_percentage(layer_ranks_all, rank_selection, foldername)

    # decompose layers
    with Bar(f'Model being decomposed with {schema} decomposition ...') as bar:
        for index in target_layers:
            new_layers[index] = decompose_conv2d_layer(
                            original_model,
                            index,
                            rank=layer_rank[index],
                            param=param,
                            schema=schema
                            )
            bar.next((100/layer_num))

    return new_layers


def insert_layer_nonseq(
        model,
        layer_regexes,
        insert_layer_factory,
        weight_map=None,
        position='replace',
        method="VBMF"
        ):
    '''
    This function is used to insert layers into model

    Key arguments:
    model -- model to be modified
    layer_regexes -- the index of layers to be modified, given in a list format
    insert_layer_factory -- the new layers list used to replace original layer
    position -- choose where to insert new layers

    Return:
    Modified model
    '''
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}
    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if(layer.name not in network_dict['input_layers_of'][layer_name]):
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [
                    network_dict['new_output_tensor_of'][layer_aux]
                    for layer_aux in network_dict['input_layers_of'][layer.name]]

        if len(layer_input) == 1:
            layer_input = layer_input[0]

        match_flag = 0
        # Insert layer if name matches the regular expression
        for layer_regex in layer_regexes:
            if (model.layers[layer_regex] == layer):
                match_flag = 1
                if position == 'replace':
                    x = layer_input
                elif position == 'after':
                    x = layer(layer_input)
                elif position == 'before':
                    pass
                else:
                    raise ValueError('position must be: before, after or replace')
                if method == "depthwise_pd":
                    [p_layers, d_layers] = insert_layer_factory[layer_regex]
                    item_number = len(p_layers)
                    ys = []
                    for i in range(item_number):
                        y = p_layers[i](x)
                        y = d_layers[i](y)
                        ys.append(y)
                    if item_number > 1:
                        x = tf.keras.layers.Add()(ys)
                    else:
                        x = ys[0]
                elif method == "depthwise_dp":
                    [p_layers, d_layers] = insert_layer_factory[layer_regex]
                    item_number = len(p_layers)
                    ys = []
                    for i in range(item_number):
                        y = d_layers[i](x)
                        y = p_layers[i](y)
                        ys.append(y)
                    if item_number > 1:
                        x = tf.keras.layers.Add()(ys)
                    else:
                        x = ys[0]
                else:
                    new_layers = insert_layer_factory[layer_regex]
                    for index, _layer in enumerate(new_layers):
                        x = _layer(x)
                        if weight_map is not None:
                            if index != len(new_layers)-1:
                                _layer.set_weights([weight_map[layer_regex][index]])
                            else:
                                if _layer.use_bias:
                                    _layer.set_weights([weight_map[layer_regex][index], weight_map[layer_regex][index+1]])
                                else:
                                    _layer.set_weights([weight_map[layer_regex][index]])
                break

        if match_flag == 0:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)


def model_decompose(
        raw_model,
        foldername,
        schema,
        rank_selection,
        option="CL",
        min_index=1,
        max_index=1,
        param=0.1,
        big_kernel_only=False
        ):
    '''
    decompose model

    Key arguments:
    raw_model -- model to be decomposed
    foldername -- folder used to save related logging and pictures
    schema -- decomposition schemes
    rank_selection -- Methods for rank estimation, the following
    param -- parameter necessary for rank estimation methods
    min_index -- only decompose the layers whose id is behind min_index
    max_index -- only decompose the layers whose id is before min_index
    option -- "CL"-> only decompose convolutional layers
              "FL"-> only decompose fully connected layers
              "CLFL"-> decompose convolutional layers as well as
                     fully connected layers
    big_kernel_only -- if only decompose layer with large receptive field

    Return:
    new_layers -- decomposed layers in a dictionary format
    '''    
    new_layers = decompose_layers(
            raw_model,
            foldername,
            schema=schema,
            rank_selection=rank_selection,
            min_index=min_index,
            max_index=max_index,
            param=param,
            big_kernel_only=big_kernel_only,
            option=option
    )

    modified_model = insert_layer_nonseq(
                raw_model,
                new_layers.keys(),
                new_layers,
                method=schema,
    )
    plot_model_decompose(modified_model, foldername)

    return modified_model
