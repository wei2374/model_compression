import tensorly as tl
import tensorflow as tf
import numpy as np
import math


def estimate_ranks_param(layer, factor=0.2, schema="tucker2D"):
    '''
    Param rank estimation
    Key arguments:
    layer -- layer to be decomposed
    factor -- how many FLOPS in convolutional layer left
    Return:
    max_rank -- ranks estimated
    '''

    tl.set_backend('tensorflow')
    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

    if schema == "channel_output":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*spatial_size*input_channel+output_channel))
        return N, N

    if schema == "depthwise_pd":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = math.ceil((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*spatial_size*output_channel+input_channel*output_channel))
        return N, N

    if schema == "VH":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*input_channel+spatial_size*output_channel))
        return N, N

    if schema == "CP":
        output_channel = layer_data.shape[0]
        input_channel = layer_data.shape[1]
        spatial_size = layer_data.shape[2]
        N = int((spatial_size*spatial_size*input_channel*output_channel*factor)/(
            spatial_size*2+output_channel+input_channel))
        return N, N

    if schema == "tucker2D" or schema == "channel_all":
        min_rank = 2
        min_rank = int(min_rank)

        initial_count = np.prod(layer_data.shape)

        cout, cin, kh, kw = layer_data.shape

        beta = max(0.8*(cout/cin), 1.)
        rate = 1./factor

        a = (beta*kh*kw)
        b = (cin + beta*cout)
        c = -initial_count/rate

        discr = b**2 - 4*a*c
        max_rank = int((-b + np.sqrt(discr))/2/a)
        # [R4, R3]
        max1 = layer_data.shape[1]*layer_data.shape[2]*layer_data.shape[3]

        max_rank = max(max_rank, min_rank)
        max_rank = [int(beta*max_rank), max_rank]
        if max_rank[0] > max1:
            max_rank[0] = max1

        max_rank = (max_rank[0], max_rank[1])

        return max_rank
    else:
        raise NotImplementedError
