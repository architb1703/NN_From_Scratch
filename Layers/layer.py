from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def __call__(self, X):
        pass

    @abstractmethod
    def grad_calc(self, grads):
        pass

    @abstractmethod
    def step(self, *args):
        pass