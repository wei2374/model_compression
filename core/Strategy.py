from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def run(self, model, config):
        self.config = config
        return
