import tensorflow as tf
from tensorflow.keras.models import Model
from decompose.decompose_schemes.decomposition_methods import decompose_conv2d_layer
from decompose.rank_estimation.rank_estimator import estimate_rank
from tools.visualization.decompose_visualization_tool import plot_rank_percentage
from tools.progress.bar import Bar


def decompose_layers(
        original_model,
        foldername,
        schema="tucker2D",
        min_index=0,
        max_index=0,
        rank_selection="VBMF",
        option="CL",
        param=0.2,
        big_kernel_only=False
        ):
    '''
    decompose layers in the given model according to option

    Keyword arguments:
    original_model -- model to be decomposed
    method -- method used to decompose layers
    min_index -- only decompose the layers after {min_index}
    decompose_time -- if the model is decomposed for the first time
                     (iterative decomposition method)
    rank_selection -- Methods for redundancy estimation, the following
                      methods are implemented
                      VBMF: VBMF rank estimation
                      weak_VNMF: VBMF rank estimation without full
                                redundancy reduction
                      BayesOpt: Bayes estimation for redundancies
                      Param: reduce ranks for flops redunction
    option -- "CL"-> only decompose convolutional layers
              "FL"-> only decompose fully connected layers
              "CLFL"-> decompose convolutional layers as well as
                     fully connected layers

    Return:
    new_layers -- decomposed layers in a dictionary format "{index of layer}:
                [decomposed layers in a list]"
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
            bar.next(int(100/layer_num))
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
            bar.next(int(100/layer_num))

    return new_layers


def insert_layer_nonseq(
        model,
        layer_regexes,
        insert_layer_factory,
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
                else:
                    new_layers = insert_layer_factory[layer_regex]
                    for _layer in new_layers:
                        x = _layer(x)
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

    return modified_model
