"""
    GoogLeNet/InceptionNet-V1 model.
"""
import torch
from torch import nn


class GoogLeNet(nn.Module):
    """
    GoogLeNet/InceptionNet-V1 model.

    Expects input of shape (3, 224, 224).
    """

    def __init__(self, num_classes: int = 100) -> None:
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
