import tensorflow as tf
from tools.visualization.model_visualization import visualize_model
from compression_tools.pruning.helper_functions import load_model_param


def build_pruned_model(original_model, new_model_param):
    layer_types, _, _, _, _ = load_model_param(original_model)
    model_config = original_model.get_config()
    prune_ratio = {}
    for i in range(0, len(model_config['layers'])):
        if model_config['layers'][i]['class_name'] == "Dense":
            model_config['layers'][i]['config']['units'] = new_model_param[i][0].shape[1]

        elif model_config['layers'][i]['class_name'] == "Conv2D":
            channel_output = model_config['layers'][i]['config']['filters']
            prune_ratio[i] = 1 - new_model_param[i][0].shape[3]/channel_output
            model_config['layers'][i]['config']['filters'] = new_model_param[i][0].shape[3]
            if (model_config['layers'][i+2]['class_name']) == "ReLU":
                model_config['layers'][i]['config']['trainable'] = True

    loss = original_model.loss
    optimizer = original_model.optimizer
    metrics = []
    if len(original_model.metrics)!=0:
        metrics.append(original_model.metrics[-1])
    
    original_model = tf.keras.Model().from_config(model_config)
    original_model.build(input_shape=original_model.input_shape)

    for i in range(0, len(original_model.layers)):
        if layer_types[i] == 'Conv2D' or\
           layer_types[i] == 'Dense'\
           or layer_types[i] == 'BatchNormalization' or\
           layer_types[i] == 'DepthwiseConv2D' and i > 0:
            original_model.layers[i].set_weights(new_model_param[i])
    # if foldername is not None:
    #     visualize_model(original_model, foldername, prune_ratio)
    original_model.compile(metrics=metrics, loss=loss, optimizer=optimizer)
    return original_model