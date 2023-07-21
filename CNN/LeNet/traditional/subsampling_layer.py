"""
    The subsampling layer used in LeNet-5.
"""
import math

import torch
import torch.nn as nn


class SubsamplingLayer(nn.Module):
    """
    The subsampling layer as described in the LeNet-5 paper.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        weights = torch.Tensor(in_channels)
        self.weights = nn.Parameter(weights)
        bias = torch.Tensor(in_channels)
        self.bias = nn.Parameter(bias)

        # Init params
        bound = 1 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.weights, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pool = nn.functional.avg_pool2d(
            x, 2, stride=2, divisor_override=1).permute((0, 2, 3, 1))
        return nn.functional.softmax(pool * self.weights + self.bias, dim=2)\
            .permute((0, 3, 1, 2))
