"""
Sparse convolutional layer.
"""

import torch
from torch import nn


class SparseConvLayer(nn.Module):
    """
    The sparse convolutional layer (C3) described in the LeNet paper.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = 5
        self.out_channels = 16
        self.weight = nn.Parameter(
            torch.Tensor(10, 6, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(1, self.out_channels, 1, 1))
        self.mapping = [
            [0, 4, 5, 6, 9, 10, 11, 12, 14, 15],
            [0, 1, 5, 6, 7, 10, 11, 12, 13, 15],
            [0, 1, 2, 6, 7, 8, 11, 13, 14, 15],
            [1, 2, 3, 6, 7, 8, 9, 12, 14, 15],
            [2, 3, 4, 7, 8, 9, 10, 12, 13, 15],
            [3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
        ]

        # Init params
        in_channels = 10
        bound = 2.4 / in_channels
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        output = torch.zeros(
            x.size(0),
            self.out_channels,
            x.size(2) - self.kernel_size + 1,
            x.size(3) - self.kernel_size + 1,
            device=self.weight.device,
        )

        for in_channel, out_channels in enumerate(self.mapping):
            output[:, out_channels, :, :] += (
                nn.functional.conv2d(  # pylint: disable=E1102
                    x[:, in_channel, :, :].unsqueeze(1),
                    self.weight[:, in_channel, :, :].unsqueeze(1),
                )
            ) + self.bias[:, out_channels, :, :]
        return output
