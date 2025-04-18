"""
Inception-V3 model.
"""

import torch
from torch import nn


class _BasicConvBlock(nn.Module):
    """Convolution block consisting of 2D convolution, followed by batch norm then ReLU."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class _GridReduction(nn.Module):
    """Grid reduction block."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self._5x5 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[0], kernel_size=1),
            _BasicConvBlock(out_channels[0], out_channels[0], kernel_size=3, padding=1),
            _BasicConvBlock(out_channels[0], out_channels[0], kernel_size=3, stride=2),
        )

        self._3x3 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[1], kernel_size=1),
            _BasicConvBlock(out_channels[1], out_channels[1], kernel_size=3, stride=2),
        )

        self._pool = nn.MaxPool2d(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.concat((self._5x5(x), self._3x3(x), self._pool(x)), 1)


class _InceptionBlockFig5(nn.Module):
    """Inception block as described in Figure 5 of the paper."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self._5x5 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[0], kernel_size=1),
            _BasicConvBlock(out_channels[0], out_channels[0], kernel_size=3, padding=1),
            _BasicConvBlock(out_channels[0], out_channels[0], kernel_size=3, padding=1),
        )

        self._3x3 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[1], kernel_size=1),
            _BasicConvBlock(out_channels[1], out_channels[1], kernel_size=3, padding=1),
        )

        self._pool_1x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _BasicConvBlock(in_channels, out_channels[2], kernel_size=1),
        )

        self._1x1 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[3], kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.cat(
            (self._5x5(x), self._3x3(x), self._pool_1x1(x), self._1x1(x)), 1
        )


class _InceptionBlockFig6(nn.Module):
    """Inception block as described in Figure 6 of the paper."""

    def __init__(self, in_channels: int, channel_size: int) -> None:
        super().__init__()
        self._5x5 = nn.Sequential(
            _BasicConvBlock(in_channels, channel_size, kernel_size=1),
            _BasicConvBlock(
                channel_size,
                channel_size,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            _BasicConvBlock(
                channel_size,
                channel_size,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
            _BasicConvBlock(
                channel_size,
                channel_size,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            _BasicConvBlock(
                channel_size,
                192,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
        )

        self._3x3 = nn.Sequential(
            _BasicConvBlock(in_channels, channel_size, kernel_size=1),
            _BasicConvBlock(
                channel_size,
                channel_size,
                kernel_size=(1, 7),
                padding=(0, 3),
            ),
            _BasicConvBlock(
                channel_size,
                192,
                kernel_size=(7, 1),
                padding=(3, 0),
            ),
        )

        self._pool_1x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _BasicConvBlock(in_channels, 192, kernel_size=1),
        )

        self._1x1 = nn.Sequential(_BasicConvBlock(in_channels, 192, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.cat(
            (self._5x5(x), self._3x3(x), self._pool_1x1(x), self._1x1(x)), 1
        )


class _InceptionBlockFig7(nn.Module):
    """Inception block as described in Figure 7 of the paper."""

    def __init__(self, in_channels: int, out_channels: list[int]) -> None:
        super().__init__()
        self._5x5_head = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[0], kernel_size=1),
            _BasicConvBlock(
                out_channels[0],
                out_channels[0],
                kernel_size=3,
                padding=1,
            ),
        )
        self._5x5_tail_1 = _BasicConvBlock(
            out_channels[0],
            out_channels[0],
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self._5x5_tail_2 = _BasicConvBlock(
            out_channels[0],
            out_channels[0],
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self._5x5_tails = [self._5x5_tail_1, self._5x5_tail_2]

        self._3x3_head = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[1], kernel_size=1),
        )
        self._3x3_tail_1 = _BasicConvBlock(
            out_channels[1],
            out_channels[1],
            kernel_size=(1, 3),
            padding=(0, 1),
        )
        self._3x3_tail_2 = _BasicConvBlock(
            out_channels[1],
            out_channels[1],
            kernel_size=(3, 1),
            padding=(1, 0),
        )
        self._3x3_tails = [self._3x3_tail_1, self._3x3_tail_2]

        self._pool_1x1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            _BasicConvBlock(in_channels, out_channels[2], kernel_size=1),
        )

        self._1x1 = nn.Sequential(
            _BasicConvBlock(in_channels, out_channels[3], kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _5x5 = self._5x5_head(x)
        _3x3 = self._3x3_head(x)

        return torch.cat(
            (
                *[layer(_5x5) for layer in self._5x5_tails],
                *[layer(_3x3) for layer in self._3x3_tails],
                self._pool_1x1(x),
                self._1x1(x),
            ),
            1,
        )


class InceptionV3(nn.Module):
    """
    Inception V3 model.

    Expects input of shape (3, 299, 299)
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.head = nn.Sequential(  # (3, 299, 299)
            _BasicConvBlock(3, 32, kernel_size=3, stride=2),  # (32, 149, 149)
            _BasicConvBlock(32, 32, kernel_size=3),  # (32, 147, 147)
            _BasicConvBlock(32, 64, kernel_size=3, padding=1),  # (64, 147, 147)
            nn.MaxPool2d(3, 2),  # (64, 73, 73)
            _BasicConvBlock(64, 80, kernel_size=3),  # (80, 71, 71)
            _BasicConvBlock(80, 192, kernel_size=3, stride=2),  # (192, 35, 35)
            *[
                _InceptionBlockFig5(in_channels, [64, 64, 96, pool_channels])
                for in_channels, pool_channels in zip([192, 256, 288], [32, 64, 64])
            ],  # (288, 35, 35)
            _GridReduction(288, [384, 96]),  # (768, 17, 17)
            *[
                _InceptionBlockFig6(768, channel_size)
                for channel_size in range(128, 193, 16)
            ],  # (768, 17, 17)
        )

        self.tail = nn.Sequential(  # (768, 17, 17)
            _GridReduction(768, [384, 128]),  # (1280, 8, 8)
            *[
                _InceptionBlockFig7(
                    in_channels,
                    [
                        (in_channels + 384) // 8,
                        (in_channels + 384) // 8,
                        (in_channels + 384) // 4,
                        (in_channels + 384) // 4,
                    ],
                )
                for in_channels in range(1280, 1665, 384)
            ],
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

        self.aux = nn.Sequential(
            nn.AvgPool2d(5, stride=3),
            _BasicConvBlock(768, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(3200, 1024),
            nn.Linear(1024, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.head(x)

        if self.training:
            return self.aux(x), self.tail(x)
        return self.tail(x)
