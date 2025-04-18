"""The Euclidean Radial Basis Function layer."""

import os

import numpy as np
import PIL.Image
import torch
from torch import nn


class RBFLayer(nn.Module):
    """The Euclidean Radial Basis Function (RBF) layer for digits."""

    in_channels: int
    out_channels: int
    kernels: torch.Tensor

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        def load_kernel(i: int):
            image = np.array(
                PIL.Image.open(
                    os.path.join(os.path.dirname(__file__), f"RBF_Kernels/{i}_RBF.jpg")
                ).convert("L")
            )
            return ((image < 127.5) * 2 - 1).flatten()

        kernels = torch.Tensor(np.array([load_kernel(i) for i in range(10)]))
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        shape = x.size(0), self.out_channels, self.in_channels
        kernels: torch.Tensor = self.kernels
        return (
            (x.unsqueeze(1).expand(shape) - kernels.unsqueeze(0).expand(shape))
            .pow(2)
            .sum(-1)
        )
