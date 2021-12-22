import numpy as np
from MINI.Core.Evaluator import Evaluator
from MINI.Tools.progress.bar import Bar
from MINI.Tools.utils.utils import find_act

class TaylorEval(Evaluator):
    def __init__(self, engine, task):
        super().__init__(engine, task)

    def evaluate(self, model):
        train_data, _ = self.task.get_dataset()
        batches_n = 20
        # Get out the corresponding activation layers of all Conv2D layers
        target_layers = {}
        for index, layer in enumerate(model.layers):
            if self.engine.is_type(layer, 'Conv2D') and\
                not self.engine.is_type(layer, 'DepthwiseConv2D'):
                target_layers[index] = find_act(layer)

        
        with Bar(f'Channel importance estimation based on first-order taylor...') as bar:
            crits = {}
            for b in range(batches_n):
                # Get gradients
                taylor_grad, activation_outputs_dic =\
                     self.engine.calculate_taylor_gradient(model, target_layers, train_data)
            
                # Compute criterion
                for n in taylor_grad:
                    crit = np.multiply(taylor_grad[n], activation_outputs_dic[n])
                    channels = crit.shape[3]
                    inputs = crit.shape[0]
                    crit = crit.transpose(3, 0, 1, 2)
                    crit = np.abs(crit.reshape(channels, inputs, -1))
                    cirt_av = np.average(np.abs(np.average(np.abs(crit), axis=2)), axis=1)
                    if b == 0:
                        crits[n] = cirt_av/batches_n
                    else:
                        crits[n] += cirt_av/batches_n
                bar.next((100/(batches_n)))

        return crits
