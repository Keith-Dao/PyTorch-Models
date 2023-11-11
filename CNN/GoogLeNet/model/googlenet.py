"""
    GoogLeNet/InceptionNet-V1 model.
"""
import torch
from torch import nn


class InceptionBlock(nn.Module):
    """
    Inception block.
    """

    def __init__(
        self,
        in_channels: int,
        _1x1_channels: int,
        _3x3_reduction: int,
        _3x3_channels: int,
        _5x5_reduction: int,
        _5x5_channels: int,
        pool_channels: int,
    ) -> None:
        super().__init__()

        self._1x1 = nn.Sequential(
            nn.Conv2d(in_channels, _1x1_channels, 1),
            nn.ReLU(True),
        )
        self._3x3 = nn.Sequential(
            nn.Conv2d(in_channels, _3x3_reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(_3x3_reduction, _3x3_channels, 3, padding=1),
            nn.ReLU(True),
        )
        self._5x5 = nn.Sequential(
            nn.Conv2d(in_channels, _5x5_reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(_5x5_reduction, _5x5_channels, 5, padding=2),
            nn.ReLU(True),
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, 1),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.cat(
            (self._1x1(x), self._3x3(x), self._5x5(x), self.pooling(x)), 1
        )


class GoogLeNet(nn.Module):
    """
    GoogLeNet/InceptionNet-V1 model.

    Expects input of shape (3, 224, 224).
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        self.input = nn.Sequential(  # (3, 224, 224)
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # (64, 112, 112)
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),  # (64, 56, 56)
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, 3, padding=1),  # (192, 56, 56)
            nn.ReLU(True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(3, stride=2, padding=1),  # (192, 28, 28)
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.input(x)
        return x
