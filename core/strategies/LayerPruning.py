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

        #TODO::depends on config
        for raw_layer in model.layers:
            import tensorflow as tf
            if isinstance(raw_layer, tf.keras.layers.Conv2D) and\
                 not isinstance(raw_layer, tf.keras.layers.DepthwiseConv2D):
                layer = Layer(raw_layer)
                layer = layer.wrap(raw_layer)
                layer.get_index(layer_index_dic)
                filter = self.param_est(layer, model, layer_index_dic)
                self.prune(layer, filter, model_params, layer_index_dic)
                self.propagate(layer, filter, model_params, layer_index_dic)
        return self.build_pruned_model(model, model_params, layer_types)

    def param_est(self, layer, model, layer_index_dic):
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
        filter = self.get_filter(layer, filters, layer_index_dic)
        return filter


    def prune(self, layer, filter, model_params, layer_index_dic):
        layer.active_prune(filter, model_params, layer_index_dic)
        return model_params


    def get_filter(self, layer, filters, dic):
        next_layers = layer.get_next_layers()
        next_layers = layer.pass_forward(next_layers)
        
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
            prev_layers = layer.get_previous_layers()
            prev_layers = layer.pass_back(prev_layers)
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
                    
                    elif any([_layer.is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]) and\
                         _layer!=layer:
                        '''---*S--layer--*T--M
                           ---*S----*T-------  '''
                        return filters[_layer.get_index(dic)]
                    
                    elif _layer.is_branch():
                        '''B--*S--layer--*T--M
                            -----*T----------     '''
                        brach_layer = _layer
                        
                id_prev_layers = brach_layer
                id_prev_layers = layer.pass_back([id_prev_layers])
                if len(id_prev_layers)>1:
                    '''--M-*T--B--*S--layer--*T--M
                       --       -----*T----------      '''
                    for prev_layer in id_prev_layers:
                        if not prev_layer.is_branch():
                            prev_layers = layer.pass_back(prev_layers)
                            stop_layer = prev_layer[0]
                            return filters[stop_layer.get_index(dic)]


                elif any([id_prev_layers[0].is_type( _layer_type) for _layer_type in self.STOP_LAYERS]):
                    '''S-*T--B--*S--layer--*T--M
                                -----*T----------      '''
                    stop_layer = id_prev_layers[0]
                    return filters[stop_layer.get_index(dic)]
                
                
        return
    

    def propagate(self, layer, filter, model_params, layer_index_dic):
        next_layers = layer.get_next_layers()
        next_layers = layer.prune_forward(filter, model_params, next_layers, layer_index_dic) 
        
        if(len(next_layers)==0):
            '''conv2d--T*'''
            return

        elif len(next_layers)==1 and\
             any([next_layers[0].is_type(_layer_type) for _layer_type in layer.STOP_LAYERS]):
            '''conv2d--T*--S'''
            for layer_type in layer.STOP_LAYERS:
                if next_layers[0].is_type(layer_type):
                    #TODO::forward declare issue
                    if(isinstance( layer.STOP_LAYERS.get(layer_type),str)):
                        end_layer = Conv2DLayer(next_layers[0].raw_layer)
                    else:
                        end_layer = layer.STOP_LAYERS.get(layer_type)(next_layers[0].raw_layer)
                    end_layer.passive_prune(filter, model_params, layer_index_dic)
                    return


        # elif next_layers[0].is_merge():
        #     '''
        #     TODO::now only handles ResNet with 2 branches ahead situation
        #     *-conv2d-T*--Merge-*
        #     ----*-------          
        #     '''
        #     if(layer.get_previous_layers()[0].is_branch()):
        #         '''
        #         B---conv2d-T*--M-*
        #          ---conv2d-T*--          
        #         '''
        #         add_layer = next_layers[0]
        #         while(True):
        #             next_layers = add_layer.get_next_layers()
        #             while(len(next_layers)!=2):
        #                 '''
        #                 *--A--*T--*
        #                 *--      
        #                 '''
        #                 next_layers = self.pass_through(filter,model_params, next_layers)
        #                 if(len(next_layers)==0):
        #                     return

        #                 if len(next_layers)==1 and any([isinstance(next_layers[0].layer, _layer) for _layer in self.STOP_LAYERS]):
        #                     '''
        #                     *--A--*T--S
        #                     *--      
        #                     '''
        #                     for layer_type in self.STOP_LAYERS:
        #                         if isinstance(next_layers[0].layer, layer_type):
        #                             if(isinstance( self.STOP_LAYERS.get(layer_type),str)):
        #                                 stop_layer = Conv2DLayer(next_layers[0].layer, self.dic)
        #                             else:
        #                                 stop_layer = self.STOP_LAYERS.get(layer_type)(next_layers[0].layer, self.dic)
        #                             stop_layer.passive_prune(filter, model_params)
        #                             return
                            
        #                     next_layers = next_layers[0].get_next_layers()


        #             for _layer in next_layers:
        #                 '''
        #                 *--A----*
        #                 *--  ---*    
        #                 '''
        #                 if isinstance(_layer, Conv2DLayer):
        #                     _layer.passive_prune(filter, model_params)
        #                 elif isinstance(_layer.layer, Add):
        #                     add_layer = _layer

        #             if(all([isinstance(_layer, Conv2DLayer) for _layer in next_layers])):
        #                 return
                        
        #     else:
        #         '''
        #         *-conv2d-T*--A-*
        #         -----------          
        #         '''

        #         pass
        

        # elif(len(next_layers)>1):
        #     '''
        #     TODO::now only handles ResNet with 2 branches ahead situation
        #     conv2d --B----
        #               ----
        #     '''
        #     while(True):
        #         while len(next_layers)!=2:
        #             next_layers = self.pass_through(filter, model_params, next_layers)
        #             if len(next_layers)==1:
        #                 next_layers = next_layers[0].get_next_layers()
                
        #         _next_layers = next_layers
        #         for _layer in next_layers:
        #             _layer = self.pass_through(filter, model_params, [_layer])[0]
        #             if isinstance(_layer, Conv2DLayer):
        #                 '''
        #                 ---B-T*-Conv2D-*-Add
        #                     -----*-----
        #                 '''
        #                 _layer.passive_prune(filter, model_params)

        #             elif isinstance(_layer.layer, Add):
        #                 '''
        #                 ---B----T*----Add
        #                     ---------
        #                 '''
        #                 next_layers = [_layer]
                
        #         if(all([isinstance(_layer, Conv2DLayer) for _layer in _next_layers])):
        #             '''
        #             ---B-T*-Conv2D-*-Add
        #                 -T*-Conv2D-*-
        #             '''
        #             return


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