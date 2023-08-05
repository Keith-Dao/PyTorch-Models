"""
    The LeNet-5 model.
"""
import math

import torch
from torch import nn

from .rbf_layer import RBFLayer
from .rescaled_tanh_layer import RescaledTanh
from .spare_conv_layer import SparseConvLayer
from .subsampling_layer import SubsamplingLayer


class LeNet5(nn.Module):
    """
    Original LeNet-5 model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.activation = RescaledTanh(1.7159, 2 / 3)
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.s2 = SubsamplingLayer(6)
        self.c3 = SparseConvLayer()
        self.s4 = SubsamplingLayer(16)
        self.c5 = nn.Conv2d(16, 120, (5, 5))
        self.f6 = nn.Linear(120, 84)
        self.out = RBFLayer(84, 10)

        # Init weights for the conv layers
        conv_layers = [self.c1, self.c5]
        for layer in conv_layers:
            bound = 2.4 / layer.in_channels
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.activation(self.c1(x))
        x = self.activation(self.s2(x))
        x = self.activation(self.c3(x))
        x = self.activation(self.s4(x))
        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.f6(x)
        return self.out(x)

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        The model's loss function.
        """
        return logits[targets == 1].pow(2).sum() + torch.log(
            math.exp(-0.1) + (-logits[targets == 0]).exp().sum()
        )
