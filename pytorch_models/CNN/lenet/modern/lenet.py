"""Modern interpretation of the LeNet-5 model."""

import torch
from torch import nn


class LeNet5(nn.Module):
    """Modern interpretation of the LeNet-5 model."""

    def __init__(self) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.s2 = nn.MaxPool2d((2, 2), (2, 2))
        self.c3 = nn.Conv2d(6, 16, (5, 5))
        self.s4 = nn.MaxPool2d((2, 2), (2, 2))
        self.c5 = nn.Conv2d(16, 120, (5, 5))
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(
        self, x: torch.Tensor, feature_maps: None | list[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        if feature_maps is not None:
            feature_maps.clear()

        x = self.activation(self.c1(x))
        x = self.activation(self.s2(x))
        if feature_maps is not None:
            feature_maps.append(x)

        x = self.activation(self.c3(x))
        x = self.activation(self.s4(x))
        if feature_maps is not None:
            feature_maps.append(x)

        x = self.activation(self.c5(x))
        x = torch.flatten(x, start_dim=1)
        x = self.f6(x)
        return self.out(x)
