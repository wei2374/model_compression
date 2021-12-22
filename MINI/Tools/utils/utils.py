
def find_act(raw_layer):
    from MINI.Core.Layer import Layer
    layer = Layer(raw_layer)
    _layer = layer.engine.wrap(raw_layer)(raw_layer)

    next_layers = _layer.get_next_layers()

    while len(next_layers)==1 and\
         (not any([next_layers[0].is_type(layer_type) for layer_type in _layer.ACT_LAYERS]) and\
            not next_layers[0].is_merge()):
        next_layers = next_layers[0].get_next_layers()
        
    if len(next_layers)==1 and any([next_layers[0].is_type(layer_type) for layer_type in _layer.ACT_LAYERS]):
        return next_layers[0].raw_layer
    
    if len(next_layers)==1 and next_layers[0].is_merge():
        while len(next_layers)==1 and\
            not any([next_layers[0].is_type(layer_type) for layer_type in _layer.ACT_LAYERS]):
            next_layers = next_layers[0].get_next_layers()

        return next_layers[0].raw_layer


def get_block_result(layer, Tensor_data):
    from MINI.Core.Layer import Layer
    next_layers = [Layer(layer)] #layer.get_next_layers()

    # layer followed by a single layer
    while(len(next_layers)==1 and not next_layers[0].is_merge()):
        Tensor_data = next_layers[0].execute(Tensor_data)
        next_layers = next_layers[0].get_next_layers()

    # layer is in the last stage
    if len(next_layers) == 0:
        # output = next_layers[0].execute(Tensor_data)
        return None, Tensor_data, False

    # layer is before a block
    elif len(next_layers) > 1:
        branch_outputs = []
        _next_layers = next_layers
        for _layer in _next_layers:
            __next_layers = [_layer]
            _Tensor_data = Tensor_data
            while(len(__next_layers)==1 and not __next_layers[0].is_merge()):
                _Tensor_data = __next_layers[0].execute(_Tensor_data)
                __next_layers = __next_layers[0].get_next_layers()
            branch_outputs.append(_Tensor_data)
            if __next_layers[0].is_merge():
                merge_layer = __next_layers[0]
        Tensor_data = merge_layer.execute(branch_outputs)
        next_layers = merge_layer.get_next_layers()
        return next_layers[0].raw_layer, Tensor_data, True

    # layer is before a merge
    elif len(next_layers) == 1 and next_layers[0].is_merge():
        branch_outputs = []
        branch_outputs.append(Tensor_data)
        mimic_data = Tensor_data
        branch_outputs.append(mimic_data)
        merge_layer = next_layers[0]
        Tensor_data = merge_layer.execute(branch_outputs)
        next_layers = merge_layer.get_next_layers()
        return next_layers[0].raw_layer, Tensor_data, True

    