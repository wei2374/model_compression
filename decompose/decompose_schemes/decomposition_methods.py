from .tensor_decomposition_tucker2D import tucker_decomposition_conv_layer
from .tensor_decomposition_cp import cp_decomposition_conv_layer
from .channel_decomposition_VH import VH_decomposition_conv_layer
from .channel_decomposition_whole import channel_decomposition_all_conv_layer
from .channel_decomposition_output import output_channel_decomposition_conv_layer
from .channel_decomposition_input import input_channel_decomposition_conv_layer
from .channel_decomposition_output_nl import channel_decomposition_nl_conv_layer
from .network_decouple_dp import network_decouple_conv_layer_dp
from .network_decouple_pd import network_decouple_conv_layer_pd


def decompose_conv2d_layer(
        original_model,
        index,
        param,
        schema,
        rank=None,
        ):
    '''
    decompose convolutional layers in the given model according to configuration

    Keyword arguments:
    original_model -- model to be decomposed
    index -- index of layer
    param -- configuration for sparsity estimation methods
    schema -- decomposition method
    rank -- sparsifity estimation parameter

    Return:
    new_layers -- decomposed layers in a dictionary format "{index of layer}:
                [decomposed layers in a list]"
    '''

    layers = [original_model.layers[index]]

    if schema == "tucker2D":
        new_layers = tucker_decomposition_conv_layer(
                        layers,
                        rank=rank
                        )
    elif schema == "CP":
        new_layers = cp_decomposition_conv_layer(
                        layers,
                        rank=rank,
                        )
    elif schema == "VH":
        new_layers = VH_decomposition_conv_layer(
                        layers,
                        rank=rank
                        )
    elif schema == "channel_all":
        new_layers = channel_decomposition_all_conv_layer(
                        layers,
                        ranks=rank,
                        )
    elif schema == "channel_output":
        new_layers = output_channel_decomposition_conv_layer(
                        layers,
                        rank=rank,
                        )
    elif schema == "channel_input":
        new_layers = input_channel_decomposition_conv_layer(
                        layers,
                        rank=rank,
                        )
    elif schema == "channel_nl":
        new_layers = channel_decomposition_nl_conv_layer(
                        original_model,
                        index,
                        layers,
                        rank=rank,
                        )
    elif schema == "depthwise_pd":
        new_layers = network_decouple_conv_layer_pd(
                        layers,
                        param=param,
                        rank=rank,
                        )
    elif schema == "depthwise_dp":
        new_layers = network_decouple_conv_layer_dp(
                        layers,
                        param=param,
                        rank=rank,
                        )
    # TODO:: add support for TT decomposition
    # elif method == "TT":
    else:
        raise NotImplementedError
    return new_layers
