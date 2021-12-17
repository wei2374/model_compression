import tensorly as tl
import tensorflow as tf
import numpy as np
import GPyOpt
from . import BayesOpt


def estimate_ranks_BayesOpt(layer):
    '''
    BayesOpt rank estimation

    Key arguments:
    layer -- layer to be decomposed

    Return:
    ranks -- ranks estimated
    '''

    func = BayesOpt.BayesOpt_rank_selection(layer)

    weights = np.asarray(layer.get_weights()[0])
    layer_data = tl.tensor(weights)
    layer_data = tf.transpose(layer_data, [3, 2, 0, 1])

    axis_0 = layer_data.shape[0]
    axis_1 = layer_data.shape[1]

    space = [
        {"name": "rank_1", "type": "continuous", "domain": (2, axis_0 - 1)},
        {"name": "rank_2", "type": "continuous", "domain": (2, axis_1 - 1)},
    ]

    feasible_region = GPyOpt.Design_space(space=space)

    initial_design = GPyOpt.experiment_design.initial_design(
        "random", feasible_region, 10
    )

    objective = GPyOpt.core.task.SingleObjective(func.f)

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False)

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(
        model, feasible_region, optimizer=acquisition_optimizer
    )

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(
        model, feasible_region, objective, acquisition, evaluator, initial_design
    )

    max_time = None
    tolerance = 10e-3
    max_iter = 10
    bo.run_optimization(
        max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=True
    )

    # bo.plot_acquisition()
    # bo.plot_convergence()

    rank1 = int(bo.x_opt[0])
    rank2 = int(bo.x_opt[1])
    ranks = [rank1, rank2]

    return ranks
