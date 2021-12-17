from tensorflow.python.keras.layers.normalization import BatchNormalization
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Conv2D, DepthwiseConv2D,\
    BatchNormalization, ZeroPadding2D, Add, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D

from core.Layer import Layer
from core.layers.BNLayer import BNLayer
from core.layers.DenseLayer import DenseLayer
from core.layers.FlattenLayer import FlattenLayer
from core.layers.DenseLayer import DenseLayer
from core.layers.DepthwiseLayer import DepthwiseLayer
from core.layers.FlattenLayer import FlattenLayer

class Conv2DLayer(Layer):
    THROUGH_LAYERS = {
    BatchNormalization: BNLayer,
    DepthwiseConv2D: DepthwiseLayer,
    Flatten: FlattenLayer,
    ZeroPadding2D: Layer,
    Activation: Layer,
    Dropout: Layer,
    AveragePooling2D: Layer,
    GlobalAveragePooling2D: Layer,
    MaxPooling2D: Layer,
    }

    STOP_LAYERS = {
        Conv2D: 'Conv2DLayer',
        Dense: DenseLayer,
    }

    def active_prune(self, filter, model_params):
        if not self.soft_prune:
            model_params[self.index][0] = np.delete(model_params[self.index][0], filter, axis=3)
            if self.layer.use_bias:
                model_params[self.index][1] = np.delete(model_params[self.index][1], filter, axis=0)
        else:
            model_params[self.index][0][:, :, :, filter] = float(0)
            if self.layer.use_bias:
                model_params[self.index][1][filter] = float(0)


    def passive_prune(self, filter, model_params):
        if not self.soft_prune:
            model_params[self.index][0] = np.delete(model_params[self.index][0], filter, axis=2)


    def find_filter(self, filters):
        next_layers = self.get_next_layers()
        while(len(next_layers)==1 and\
            any([isinstance(next_layers[0].layer, _layer) for _layer in self.THROUGH_LAYERS])):
            next_layers = next_layers[0].get_next_layers()

        if(len(next_layers)==0):
            return filters[self.index]

        if(len(next_layers)>1):
            return filters[self.index]
        
        if(any([isinstance(next_layers[0].layer, _layer) for _layer in self.STOP_LAYERS])):
            return filters[self.index]

        if(isinstance(next_layers[0].layer, Add)):
            prev_layers = self.get_previous_layers()
            if len(prev_layers)==1 and len(prev_layers[0].get_next_layers())==2:
                return filters[self.index]
            else:
                prev_layers = next_layers[0].get_previous_layers()
                for layer in prev_layers:
                    _layer = layer
                    while(len(_layer.get_next_layers())!=2 and not isinstance(_layer, Conv2DLayer)):
                        _layer = _layer.get_previous_layers()[0]
                    if(len(layer.get_next_layers())==2):
                        identical_b_end = layer
                    elif(isinstance(_layer, Conv2DLayer) and _layer!=self):
                        return filters[_layer.index]
                        
                id_prev_layers = identical_b_end
                if len(id_prev_layers.get_next_layers())!=2:
                    prev_layer = id_prev_layers
                    while(not isinstance(prev_layer, Conv2D)):
                        prev_layer = prev_layer.get_previous_layers()[0]
                    id_conv = prev_layer
                    return filters[id_conv.index]
                
                elif len(id_prev_layers.get_next_layers())==2:
                    _layer = id_prev_layers
                    while(len(_layer.get_previous_layers())==1):
                        if(isinstance(_layer, Conv2DLayer)):
                            return filters[_layer.index]
                        _layer = _layer.get_previous_layers()[0]
                    add_layer = _layer

                    for prev_layer in add_layer.get_previous_layers():
                        if len(prev_layer.get_next_layers())!=2:
                            while(not isinstance(prev_layer, Conv2D)):
                                prev_layer = prev_layer.get_previous_layers()[0]
                            id_conv = prev_layer
                            return filters[id_conv.index]


    def pass_through(self, filter, model_params, next_layers):
        while(len(next_layers)==1 and\
            any([isinstance(next_layers[0].layer, _layer) for _layer in self.THROUGH_LAYERS])):
            for layer_type in self.THROUGH_LAYERS:
                if isinstance(next_layers[0].layer, layer_type):
                    through_layer = self.THROUGH_LAYERS.get(layer_type)(next_layers[0].layer, self.dic)
                    through_layer.passive_prune(filter, model_params)
                    next_layers = [through_layer]
                    break
            next_layers = next_layers[0].get_next_layers()
        
        return next_layers

    def propagate(self, filter, model_params):
        next_layers = self.get_next_layers()
        next_layers = self.pass_through(filter, model_params, next_layers)
        
        if(len(next_layers)==0):
            '''conv2d--T*'''
            return


        elif len(next_layers)==1 and any([isinstance(next_layers[0].layer, _layer) for _layer in self.STOP_LAYERS]):
            '''conv2d--T*--S'''
            for layer_type in self.STOP_LAYERS:
                if isinstance(next_layers[0].layer, layer_type):
                    if(isinstance( self.STOP_LAYERS.get(layer_type),str)):
                        end_layer = Conv2DLayer(next_layers[0].layer, self.dic)
                    else:
                        end_layer = self.STOP_LAYERS.get(layer_type)(next_layers[0].layer, self.dic)
                    end_layer.passive_prune(filter, model_params)
            return


        elif(isinstance(next_layers[0].layer, Add)):
            '''
            TODO::now only handles ResNet with 2 branches ahead situation
            *-conv2d-T*--A-*
            ----*-------          
            '''
            if(len(self.get_previous_layers()[0].get_next_layers())==2):
                '''
                *-conv2d-T*--A-*
                *-conv2d-T*--          
                '''
                add_layer = next_layers[0]
                while(True):
                    next_layers = add_layer.get_next_layers()
                    while(len(next_layers)!=2):
                        '''
                        *--A--*T--*
                        *--      
                        '''
                        next_layers = self.pass_through(filter,model_params, next_layers)
                        if(len(next_layers)==0):
                            return

                        if len(next_layers)==1 and any([isinstance(next_layers[0].layer, _layer) for _layer in self.STOP_LAYERS]):
                            '''
                            *--A--*T--S
                            *--      
                            '''
                            for layer_type in self.STOP_LAYERS:
                                if isinstance(next_layers[0].layer, layer_type):
                                    if(isinstance( self.STOP_LAYERS.get(layer_type),str)):
                                        stop_layer = Conv2DLayer(next_layers[0].layer, self.dic)
                                    else:
                                        stop_layer = self.STOP_LAYERS.get(layer_type)(next_layers[0].layer, self.dic)
                                    stop_layer.passive_prune(filter, model_params)
                                    return
                            
                            next_layers = next_layers[0].get_next_layers()


                    for _layer in next_layers:
                        '''
                        *--A----*
                        *--  ---*    
                        '''
                        if isinstance(_layer, Conv2DLayer):
                            _layer.passive_prune(filter, model_params)
                        elif isinstance(_layer.layer, Add):
                            add_layer = _layer

                    if(all([isinstance(_layer, Conv2DLayer) for _layer in next_layers])):
                        return
                        
            else:
                '''
                *-conv2d-T*--A-*
                -----------          
                '''

                pass
        

        elif(len(next_layers)>1):
            '''
            TODO::now only handles ResNet with 2 branches ahead situation
            conv2d --B----
                      ----
            '''
            while(True):
                while len(next_layers)!=2:
                    next_layers = self.pass_through(filter, model_params, next_layers)
                    if len(next_layers)==1:
                        next_layers = next_layers[0].get_next_layers()
                
                _next_layers = next_layers
                for _layer in next_layers:
                    _layer = self.pass_through(filter, model_params, [_layer])[0]
                    if isinstance(_layer, Conv2DLayer):
                        '''
                        ---B-T*-Conv2D-*-Add
                            -----*-----
                        '''
                        _layer.passive_prune(filter, model_params)

                    elif isinstance(_layer.layer, Add):
                        '''
                        ---B----T*----Add
                            ---------
                        '''
                        next_layers = [_layer]
                
                if(all([isinstance(_layer, Conv2DLayer) for _layer in _next_layers])):
                    '''
                    ---B-T*-Conv2D-*-Add
                        -T*-Conv2D-*-
                    '''
                    return
        
