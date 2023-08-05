"""
    A rescaled tanh function.
"""
import torch
from torch import nn


class RescaledTanh(nn.Tanh):
    """
    A rescaled tanh function in the form f(a) = A tanh(Sa).
    """

    def __init__(self, A: float, S: float):
        super().__init__()
        self.A = A
        self.S = S

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.A * super().forward(self.S * input)
