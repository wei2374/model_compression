from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, task) -> None:
        self.task = task

    @abstractmethod
    def run(self, model, config):
        self.config = config
        return
