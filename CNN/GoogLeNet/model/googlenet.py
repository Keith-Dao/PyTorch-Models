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

        self.inception_3a_to_3b = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),  # (256, 28, 28)
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),  # (480, 28, 28)
        )
        self.max_pool_3 = nn.MaxPool2d(3, stride=2, padding=1)  # (480, 14, 14)

        self.inception_4a = InceptionBlock(
            480, 192, 96, 208, 16, 48, 64
        )  # (512, 14, 14)
        self.aux_1 = nn.Sequential(
            nn.AvgPool2d(5, stride=3),  # (512, 4, 4)
            nn.Conv2d(512, 128, 1),  # (128, 4, 4),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.inception_4b_to_4d = nn.Sequential(
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),  # (512, 14, 14)
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),  # (512, 14, 14)
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),  # (528, 14, 14)
        )
        self.aux_2 = nn.Sequential(
            nn.AvgPool2d(5, stride=3),  # (528, 4, 4)
            nn.Conv2d(528, 128, 1),  # (128, 4, 4),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.inception_4e = InceptionBlock(
            528, 256, 160, 320, 32, 128, 128
        )  # (832, 14, 14)
        self.max_pool_4 = nn.MaxPool2d(3, stride=2, padding=1)  # (832, 7, 7)

        self.inception_5a_to_5b = nn.Sequential(
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),  # (832, 7, 7)
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),  # (1024, 7, 7)
        )

        self.output = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.input(x)

        x = self.inception_3a_to_3b(x)
        x = self.max_pool_3(x)

        x = self.inception_4a(x)
        if self.training:
            aux_1 = self.aux_1(x)
        x = self.inception_4b_to_4d(x)
        if self.training:
            aux_2 = self.aux_2(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)

        x = self.inception_5a_to_5b(x)
        x = self.output(x)

        if self.training:
            return x, aux_1, aux_2  # type: ignore
        return x
