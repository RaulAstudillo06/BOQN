import torch
from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod

class ObjectiveFunction(Module, ABC):
    def __init__(self, objective_function):
        self.objective_function = objective_function
        self.X_pending = None
        
    def forward(self, X: Tensor) -> Tensor:
        return self.objective_function(X)

