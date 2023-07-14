"""
    The LeNet-5 model.
"""
import torch
import torch.nn as nn

from subsampling_layer import SubsamplingLayer


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.s2 = SubsamplingLayer(6)
        self.c3 = nn.Conv2d(6, 16, (5, 5))
        self.s4 = SubsamplingLayer(16)
        self.c5 = nn.Conv2d(16, 120, (5, 5))
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.activation(self.c1(x))
        x = self.s2(x)
        x = self.activation(self.c3(x))
        x = self.s4(x)
        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.f6(x)
        return self.out(x)
