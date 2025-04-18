"""The AlexNet model."""

import torch
from torch import nn


class AlexNet(nn.Module):
    """AlexNet model."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.net = nn.Sequential(  # (3, 227, 227)
            nn.Conv2d(3, 96, (11, 11), 4),  # (96, 55, 55)
            nn.ReLU(),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2)),  # (96, 27, 27)
            nn.Conv2d(96, 256, (5, 5), padding=(2, 2)),  # (256, 27, 27)
            nn.ReLU(),
            nn.LocalResponseNorm(5, 1e-4, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2)),  # (256, 13, 13)
            nn.Conv2d(256, 384, (3, 3), padding=(1, 1)),  # (384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, (3, 3), padding=(1, 1)),  # (384, 13, 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), padding=(1, 1)),  # (256, 13, 13)
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2)),  # (256, 6, 6)
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)
