"""
    The LeNet-5 model.
"""
import math

import torch
import torch.nn as nn

from subsampling_layer import SubsamplingLayer


class SparseConvLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = 5
        self.out_channels = 16
        self.weights = nn.Parameter(torch.Tensor(
            10, 6, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        self.mapping = [
            [0, 4, 5, 6, 9, 10, 11, 12, 14, 15],
            [0, 1, 5, 6, 7, 10, 11, 12, 13, 15],
            [0, 1, 2, 6, 7, 8, 11, 13, 14, 15],
            [1, 2, 3, 6, 7, 8, 9, 12, 14, 15],
            [2, 3, 4, 7, 8, 9, 10, 12, 13, 15],
            [3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
        ]

        # Init params
        bound = 1 / math.sqrt(6)
        nn.init.uniform_(self.weights, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            x.size(0),
            self.out_channels,
            x.size(2) - self.kernel_size + 1,
            x.size(3) - self.kernel_size + 1
        )

        for in_channel, out_channels in enumerate(self.mapping):
            output[:, out_channels, :, :] += (
                nn.functional.conv2d(
                    x[:, in_channel, :, :].unsqueeze(1),
                    self.weights[:, in_channel, :, :].unsqueeze(1)
                )
            ) + self.bias[:, out_channels, :, :]
        return output


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.s2 = SubsamplingLayer(6)
        self.c3 = SparseConvLayer()
        self.s4 = SubsamplingLayer(16)
        self.c5 = nn.Conv2d(16, 120, (5, 5))
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.activation(self.c1(x))
        x = self.activation(self.s2(x))
        x = self.activation(self.c3(x))
        x = self.activation(self.s4(x))
        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.f6(x)
        return self.out(x)
