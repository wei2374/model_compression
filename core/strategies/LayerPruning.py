import tensorflow
from core.Strategy import Strategy
from core.Layer import Layer
from core.layers.Conv2DLayer import Conv2DLayer
from core.engines.TFEngine import TFEngine
import numpy as np


class LayerPruning(Strategy):
    def __init__(self) -> None:
        super().__init__()
    
    def run(self, model, config):
        engine = TFEngine()
        layer_types, model_params, _, _, layer_index_dic = engine.load_model_param(model)
        filters = self.param_est(model)

        #TODO::depends on config
        for raw_layer in model.layers:
            import tensorflow as tf
            if isinstance(raw_layer, tf.keras.layers.Conv2D) and\
                 not isinstance(raw_layer, tf.keras.layers.DepthwiseConv2D):
                layer = Layer(raw_layer)
                layer = layer.wrap(raw_layer)
                id = layer.get_index(layer_index_dic)
                filter = self.get_filter(layer, filters, layer_index_dic)
                self.prune(layer, filter, model_params, layer_index_dic)
                self.propagate(layer, filter, model_params, layer_index_dic)
        return self.build_pruned_model(model, model_params, layer_types)

    def param_est(self, model):
        from compression_tools.pruning.pruning_criterions.magnitude import get_and_plot_weights
        score = get_and_plot_weights(model)
        #TODO::prune_ratio
        prune_ratio = 0.5
        filters = {}
        for sc in score:
            score_sorted = np.sort(score[sc])
            index = int(prune_ratio*len(score_sorted))
            prun_threshold = score_sorted[index]
            prune_mask = score[sc] <= prun_threshold
            filter = np.where(prune_mask)
            filters[sc]=filter
        return filters


    def prune(self, layer, filter, model_params, layer_index_dic):
        layer.active_prune(filter, model_params, layer_index_dic)
        return model_params


    def get_filter(self, layer, filters, dic):
        next_layers = layer.pass_forward()
        
        if(len(next_layers)==0):
            ''' layer--*T       '''
            return filters[layer.get_index(dic)]

        elif(len(next_layers)>1):
            ''' layer--*T-B--
                           --   '''
            return filters[layer.get_index(dic)]
        
        elif(any([next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS])):
            ''' layer--*T--S     '''
            return filters[layer.get_index(dic)]

        elif next_layers[0].is_merge():
            merge_layer = next_layers[0]
            #TODO::only works for ResNet
            ''' layer--*T--M
                     --*---       '''
            prev_layers = layer.pass_back()
            if len(prev_layers)>1:
                #TODO
                ''' ---
                    ---M---*T--layer--*T--M
                        --*--------------      '''
                pass

            elif prev_layers[0].is_branch():
                ''' B---*T--layer--*T--M
                     --*--------------      '''
                return filters[layer.get_index(dic)]
            else:
                '''---*S--layer--*T--M
                    --*--------------      '''
                prev_layers = merge_layer.get_previous_layers()
                for prev_layer in prev_layers:
                    _layer = prev_layer
                    _layers = layer.pass_back([_layer])
                    if len(_layers)>1:
                        #TODO
                        '''---*S--layer--*T--M
                           ---*S----*T-------  
                           ------M              '''
                        pass
                    
                    elif len(_layers)==1 and\
                        any([_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]) and\
                         _layers[0]!=layer:
                        '''---*S--layer--*T--M
                           ---*S----*T-------  '''
                        return filters[_layers[0].get_index(dic)]
                    
                    elif len(_layers)==1 and \
                        _layers[0].is_branch():
                        '''B--*S--layer--*T--M
                            -----*T----------     '''
                        branch_layer = _layer
                        
                id_prev_layers = layer.pass_back(branch_layer.get_previous_layers())
                if len(id_prev_layers)>1:
                    '''
                    *---M-*T--B--*S--layer--*T--M
                    *---       -----*T----------      
                    '''
                    for prev_layer in id_prev_layers:
                        prev_layers = layer.pass_back([prev_layer])
                        if len(prev_layers)==1 and\
                            any([prev_layers[0].is_type( _layer_type) for _layer_type in layer.STOP_LAYERS]):
                            stop_layer = prev_layer[0]
                            return filters[stop_layer.get_index(dic)]


                elif any([id_prev_layers[0].is_type( _layer_type) for _layer_type in layer.STOP_LAYERS]):
                    '''
                    S-*T--B--*S--layer--*T--M
                           -----*T----------
                    '''
                    stop_layer = id_prev_layers[0]
                    return filters[stop_layer.get_index(dic)]
                
                
        return
    

    def propagate(self, layer, filter, model_params, layer_index_dic):
        next_layers = layer.prune_forward(filter, model_params, layer_index_dic) 
        
        if(len(next_layers)==0):
            '''
            layer--*T
            '''
            return

        elif len(next_layers)==1 and\
             any([next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]):
            '''
            layer--*T--S
            '''
            for layer_type in layer.STOP_LAYERS:
                if next_layers[0].is_type(layer_type):
                    #TODO::forward declare issue
                    if(isinstance( layer.STOP_LAYERS.get(layer_type),str)):
                        end_layer = Conv2DLayer(next_layers[0].raw_layer)
                    else:
                        end_layer = layer.STOP_LAYERS.get(layer_type)(next_layers[0].raw_layer)
                    end_layer.passive_prune(filter, model_params, layer_index_dic)
                    return


        elif next_layers[0].is_merge():
            '''
            TODO::now only handles ResNet with 2 branches ahead situation
            *-layer--*T--M-*
              ----*------          
            '''
            merge_layer = next_layers[0]
            prev_layers = layer.pass_back()
            if len(prev_layers)==1 and prev_layers[0].is_branch():
                '''
                --*--B-*T-layer-*T--M-*
                      ---*----------          
                '''
                while(True):
                    next_layers = layer.prune_forward(filter, model_params, layer_index_dic, merge_layer.get_next_layers())
                    if(len(next_layers)==0):
                        '''
                        --*--B-*T-layer-*T--M-*T
                            ---*----------          
                        '''
                        return
                    elif(len(next_layers)==1 and\
                        any([next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS])):
                        '''
                        --*--B-*T-layer-*T--M-*T-S
                            ---*----------          
                        '''
                        for layer_type in layer.STOP_LAYERS:
                            if next_layers[0].is_type(layer_type):
                                if(isinstance(layer.STOP_LAYERS.get(layer_type), str)):
                                    stop_layer = Conv2DLayer(next_layers[0].raw_layer)
                                else:
                                    stop_layer = layer.STOP_LAYERS.get(layer_type)(next_layers[0].raw_layer)
                                stop_layer.passive_prune(filter, model_params, layer_index_dic)
                                return
                    else:
                        '''
                        --*--B-*T-layer-*T--M-*T-B--*
                            ---*---------         --* 
                        '''
                        old_merge_layer = merge_layer
                        for next_layer in next_layers:
                            _next_layers = layer.prune_forward(filter, model_params, layer_index_dic, [next_layer])
                            if len(_next_layers)==1 and\
                               any([_next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]):
                               for layer_type in layer.STOP_LAYERS:
                                    if _next_layers[0].is_type(layer_type):
                                        if(isinstance(layer.STOP_LAYERS.get(layer_type), str)):
                                            stop_layer = Conv2DLayer(_next_layers[0].raw_layer)
                                        else:
                                            stop_layer = layer.STOP_LAYERS.get(layer_type)(_next_layers[0].raw_layer)
                                        stop_layer.passive_prune(filter, model_params, layer_index_dic)
                        
                            elif len(_next_layers)==1 and\
                                _next_layers[0].is_merge():
                                merge_layer = _next_layers[0]

                    if(merge_layer==old_merge_layer):
                        return
            else:
                '''
                --*--S-*T-layer-*T--M-*
                -----------*---------          
                '''

                pass
        

        elif(len(next_layers)>1):
            '''
            TODO::now only handles ResNet with 2 branches ahead situation
            layer-*T--B--*
                       --*
            '''
            while(True):
                while len(next_layers)==1:
                    if any([next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]):
                        '''
                        layer-*T--B--*T--M--*T--*S
                                   --*---
                        '''
                        next_layers[0].passive_prune(filter, model_params, layer_index_dic)
                        return

                    next_layers = layer.prune_forward(filter, model_params, next_layers, next_layers[0].get_next_layers())
                
                _next_layers = next_layers
                old_next_layer = next_layers
                for _layer in next_layers:
                    _layer = layer.prune_forward(filter, model_params, layer_index_dic, [_layer])
                    if len(_layer)==1 and\
                        any([_layer[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]):
                        '''
                        layer-*T--B--*T--S
                                   --*----
                        '''
                        for layer_type in layer.STOP_LAYERS:
                            if _layer[0].is_type(layer_type):
                                # if(isinstance(layer.STOP_LAYERS.get(layer_type), str)):
                                #     stop_layer = Conv2DLayer(_layer[0].raw_layer)
                                # else:
                                #     stop_layer = layer.STOP_LAYERS.get(layer_type)(_layer[0].raw_layer)
                                _layer[0].passive_prune(filter, model_params, layer_index_dic)

                    elif len(_layer)==1 and\
                        _layer[0].is_merge():
                        '''
                        layer-*T--B--*T--M
                                   --*--
                        '''
                        next_layers = _layer
                
                if(old_next_layer==next_layers):
                    '''
                        layer-*T--B--*T-S
                                   --*T-S
                    '''
                    return


    def build_pruned_model(self, original_model, new_model_param, layer_types):
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
        
        import tensorflow as tf
        original_model = tf.keras.Model().from_config(model_config)
        original_model.build(input_shape=original_model.input_shape)

        for i in range(0, len(original_model.layers)):
            if layer_types[i] == 'Conv2D' or\
            layer_types[i] == 'Dense'\
            or layer_types[i] == 'BatchNormalization' or\
            layer_types[i] == 'DepthwiseConv2D' and i > 0:
                original_model.layers[i].set_weights(new_model_param[i])
        original_model.compile(metrics=metrics, loss=loss, optimizer=optimizer)
        return original_model