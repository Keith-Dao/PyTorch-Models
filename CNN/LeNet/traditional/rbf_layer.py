"""
    The Euclidean Radial Basis Function layer.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import PIL.Image


class RBFLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        def load_kernel(i: int):
            image = np.array(
                PIL.Image.open(
                    os.path.join(
                        os.path.dirname(__file__), f"RBF_Kernels/{i}_RBF.jpg"
                    )
                ).convert("L")
            )
            image = (image - 127) / 255.0
            return image.flatten()

        kernels = torch.Tensor(np.array([load_kernel(i) for i in range(10)]))
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size(0), self.out_channels, self.in_channels
        x = x.unsqueeze(1).expand(shape)
        kernel = self.kernels.unsqueeze(
            0
        ).expand(  # pyright: ignore [reportGeneralTypeIssues]
            shape
        )
        return (x - kernel).pow(2).sum(-1)
