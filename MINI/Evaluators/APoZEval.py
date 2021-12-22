import numpy as np
from MINI.Core.Evaluator import Evaluator
from MINI.Tools.progress.bar import Bar


class APoZEval(Evaluator):
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

        with Bar(f'Channel importance estimation based on average percentage of zero of activation output...') as bar:
            for b in range(batches_n):
                # Get the output of activation layers in the model given train_data
                activation_outputs = self.engine.get_intermediate_output(model, target_layers, train_data)
                activation_outputs_dic = {}
                for counter, layer_index in enumerate(target_layers):
                    activation_outputs_dic[layer_index] = activation_outputs[counter]
                    activation_out = np.asarray(activation_outputs_dic[layer_index])
                    in_channels = activation_out.shape[0]
                    out_channels = activation_out.shape[3]
                    activation_out = activation_out.transpose(3, 0, 1, 2).reshape(
                                        out_channels, in_channels, -1)
                    crit_mask = (activation_out > 0)
                    activation_out_mask_av2 = np.average(
                                np.sum(crit_mask, axis=2)/float(crit_mask.shape[2]), axis=1)
                    if b == 0:
                        crits[layer_index] = activation_out_mask_av2/batches_n
                    else:
                        crits[layer_index] += activation_out_mask_av2/batches_n
                bar.next(int(100/(batches_n)))
        return crits