"""
    ResNet model.
"""
import torch
from torch import nn


class BottleneckResidualBlock(nn.Module):
    """
    A bottleneck residual block.
    """

    def __init__(
        self, in_channels: int, out_channels: int, downsample: bool = False
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)

        self.residual_connection = (
            nn.Identity()
            if in_channels == out_channels and not downsample
            else nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride=2 if downsample else 1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        )

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 4,
                1,
                stride=2 if downsample else 1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 4),
            self.relu,
            nn.Conv2d(
                out_channels // 4,
                out_channels // 4,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels // 4),
            self.relu,
            nn.Conv2d(
                out_channels // 4,
                out_channels,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.relu(
            self.bottleneck_block(x) + self.residual_connection(x)
        )


class ResNet(nn.Module):
    """
    ResNet model with 50, 101 or 152 layers.
    """

    _MODEL_CONFIGURATIONS: dict[int, list[int]] = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }

    def __init__(self, num_classes: int = 1000, num_layers: int = 50) -> None:
        super().__init__()

        if num_layers not in self._MODEL_CONFIGURATIONS:
            raise ValueError(
                f"{num_layers} is not a valid configuration. Select from "
                f"""{
                    ', '.join(
                        str(config)
                        for config in self._MODEL_CONFIGURATIONS
                    )
                }"""
            )

        self.net = nn.Sequential(  # (3, 224, 224)
            nn.Conv2d(
                3, 64, 7, stride=2, padding=3, bias=False
            ),  # (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1),  # (64, 56, 56)
            *[
                nn.Sequential(
                    *[
                        BottleneckResidualBlock(
                            64
                            << block_group
                            << (block_num != 0)
                            << (block_group != 0)
                            << (block_group == 0 and block_num != 0),
                            256 << block_group,
                            downsample=block_group != 0 and block_num == 0,
                        )
                        for block_num in range(num_blocks)
                    ]
                )
                for block_group, num_blocks in enumerate(
                    self._MODEL_CONFIGURATIONS[num_layers]
                )
            ],  # (2048, 7, 7)
            nn.AvgPool2d(7),  # (2048, 1, 1)
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)
