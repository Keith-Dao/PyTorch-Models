"""
    VGG-16 Model
"""
import torch
import torch.nn as nn


class VGG(nn.Module):
    """
    VGG-16 model.
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.net = nn.Sequential(  # (3, 224, 224)
            nn.Conv2d(3, 64, 3, padding=1),  # (64, 224, 224)
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),  # (64, 224, 224)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (64, 112, 112)
            nn.Conv2d(64, 128, 3, padding=1),  # (128, 112, 112)
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),  # (128, 112, 112)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (128, 56, 56)
            nn.Conv2d(128, 256, 3, padding=1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),  # (256, 56, 56)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (256, 28, 28)
            nn.Conv2d(256, 512, 3, padding=1),  # (512, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),  # (512, 28, 28)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),  # (512, 28, 28)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (512, 14, 14)
            nn.Conv2d(512, 512, 3, padding=1),  # (512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),  # (512, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),  # (512, 14, 14),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (512, 7, 7)
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        """
        return self.net(x)
