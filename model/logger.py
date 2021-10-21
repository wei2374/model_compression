from keras_flops import get_flops


class Logger:
    def __init__(
                self,
                foldername):
        self.filename = foldername+"/log.txt"
        f = open(self.filename, "w")
        f.close()

    def log_model(self, model):
        flops = get_flops(model, batch_size=1)
        f = open(self.filename, "a")
        f.write(f"model has {flops} FLOPS\n")
        f.close()

    def log_decomposition(self, decomposition_setting):
        f = open(self.filename, "a")
        f.write(f"Channel settings:\n")
        f.write(f"Decomposition schema is {decomposition_setting['schema']}\n")
        f.write(f"Rank estimation method is {decomposition_setting['rank_selection']}\n")
        f.write(f"param is {decomposition_setting['param']}\n")
        f.write(f"Start with layer index {decomposition_setting['range'][0]}\n")
        f.write(f"End with layer index {decomposition_setting['range'][1]}\n")
        f.close()

    def log_pruning(self, pruning_settings):
        f = open(self.filename, "a")
        f.write(f"Channel pruning settings:\n")
        f.write(f"Pruning method is {pruning_settings['method']}\n")
        f.write(f"Prune ratio estimation method is {pruning_settings['ratio_est']}\n")
        f.write(f"channel estimation method is {pruning_settings['criterion']}\n")
        f.write(f"param is {pruning_settings['param']}\n")
        f.write(f"Start with layer index {pruning_settings['range'][0]}\n")
        f.write(f"End with layer index {pruning_settings['range'][1]}\n")
        f.close()

    def writeln(self, line):
        f = open(self.filename, "a")
        f.write(line+"\n")
        f.close()
