from core.layers.Conv2DLayer import Conv2DLayer

def delete_filter_after(
        new_model_param,
        layer_index,
        layer_index_dic,
        filters,
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
    _layer = Conv2DLayer(layer_index_dic[layer_index], layer_index_dic)
    filter = _layer.find_filter(filters)
    _layer.active_prune(filter, new_model_param)
    _layer.propagate(filter, new_model_param)
    return new_model_param