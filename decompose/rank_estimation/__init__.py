import tensorflow as tf
import random
from .rank_estimate_bayesopt import estimate_ranks_BayesOpt
from .rank_estimate_energy import estimate_ranks_energy
from .rank_estimate_param import estimate_ranks_param
from .rank_estimate_vbmf import estimate_ranks_VBMF


schema_rank_est_map = {}
schema_rank_est_map["tucker2D"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["channel_all"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["channel_output"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["channel_input"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["channel_nl"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["channel_nl"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["depthwise_pd"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["depthwise_dp"] = ["random", "Param", "VBMF_auto", "VBMF", "BayesOpt", "energy"]
schema_rank_est_map["VH"] = ["random", "Param", "VBMF_auto", "VBMF", "energy"]
schema_rank_est_map["CP"] = ["random", "Param"]


def estimate_rank(layer, rank_selection, param, schema):
    if rank_selection == "random":
        ranks = [int(
            random.uniform(0.1, 0.75)*layer.output_shape[3]), int(
                random.uniform(0.2, 0.5)*layer.input_shape[3])]
    
    elif rank_selection == "Param":
        ranks = estimate_ranks_param(layer, param, schema)

    elif rank_selection == "VBMF_auto" and "VBMF_auto" in schema_rank_est_map[schema]:
        ranks = estimate_ranks_VBMF(layer, factor=None, schema=schema)

    elif rank_selection == "VBMF" and "VBMF" in schema_rank_est_map[schema]:
        ranks = estimate_ranks_VBMF(layer, factor=param, schema=schema)

    elif rank_selection == "BayesOpt" and "BayesOpt" in schema_rank_est_map[schema]:
            ranks = estimate_ranks_BayesOpt(layer)
    
    elif rank_selection == "energy" and "energy" in schema_rank_est_map[schema]:
        ranks = estimate_ranks_energy(layer, param)

    else:
        print(f"Unknown or not compatible rank_selection method {rank_selection}")
        raise NotImplementedError
    return ranks