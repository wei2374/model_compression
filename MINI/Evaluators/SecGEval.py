import numpy as np
from MINI.Core.Evaluator import Evaluator
from MINI.Tools.progress.bar import Bar


class SecGEval(Evaluator):
    def evaluate(self, model):
        train_data, _ = self.task.get_dataset()
        batches_n = 20
        crits = {}
        # Get out the weights of all Conv2D layers
        target_layers = {}
        for index, layer in enumerate(model.layers):
            if self.engine.is_type(layer, 'Conv2D') and\
                not self.engine.is_type(layer, 'DepthwiseConv2D'):
                target_layers[index] = layer

        with Bar(f'Channel importance estimation based on second-order gradient...') as bar:
            crits = {}
            for b in range(batches_n):
                # Get gradients
                sec_grad = self.engine.calculate_sec_gradient(model, target_layers, train_data)
            
                # Compute criterion
                for n in sec_grad:
                    crit = np.multiply(sec_grad[n], self.engine.get_weights(target_layers[n])[0])
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
