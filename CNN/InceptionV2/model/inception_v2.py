"""
    Inception-V2 model.
"""
import torch
from torch import nn


class BatchNorm2D(nn.Module):
    """
    Batch normalization layer for 2D inputs with an additional channel dimension.
    """

    def __init__(
        self, in_channels: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels))
        self.bias = nn.Parameter(torch.empty(in_channels))
        self.eps = eps

        self.momentum = momentum  # Use exponential moving average
        self.register_buffer(
            "running_mean", torch.empty(in_channels, requires_grad=False)
        )
        self.running_mean: torch.Tensor
        self.register_buffer(
            "running_variance", torch.empty(in_channels, requires_grad=False)
        )
        self.running_variance: torch.Tensor

        self.reset_parameters()

    def reset_tracked_stats(self) -> None:
        """Resets tracked stats."""
        self.running_mean.zero_()
        self.running_variance.fill_(1)

    def reset_parameters(self) -> None:
        """Resets layer parameters."""
        self.reset_tracked_stats()
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if len(x.shape) != 4:
            raise ValueError(
                f"Expected input of dimension 4, got {len(x.shape)}."
            )

        if x.shape[1] != len(self.weight):
            raise ValueError(
                f"Expected dimension 1 of input to be {len(self.weight)}, got {x.shape[1]}."
            )

        if self.training:
            mean = x.mean([0, 2, 3])
            variance = x.var([0, 2, 3], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = (
                    self.momentum * mean
                    + (1 - self.momentum) * self.running_mean
                )
                self.running_variance = (
                    self.momentum * variance * n / (n - 1)
                    + (1 - self.momentum) * self.running_variance
                )
        else:
            mean = self.running_mean
            variance = self.running_variance

        return (
            self.weight[None, :, None, None]
            * (x - mean[None, :, None, None])
            / torch.sqrt(variance[None, :, None, None] + self.eps)
            + self.bias[None, :, None, None]
        )


class BasicConv2DBlock(nn.Module):
    """
    Basic convolutional block with 2d conv, followed by batch norm then ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            BatchNorm2D(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class InceptionBlock(nn.Module):
    """
    Inception block.
    """

    MAX_POOL = nn.MaxPool2d
    AVG_POOL = nn.AvgPool2d

    def __init__(
        self,
        in_channels: int,
        _1x1_channels: int,
        _3x3_reduction: int,
        _3x3_channels: int,
        _5x5_reduction: int,
        _5x5_channels: int,
        pooling_layer: type[nn.MaxPool2d] | type[nn.AvgPool2d],
        pool_channels: int | None,
        final_stride: int = 1,
    ) -> None:
        super().__init__()

        self._1x1 = (
            BasicConv2DBlock(
                in_channels,
                _1x1_channels,
                kernel_size=1,
                stride=final_stride,
            )
            if _1x1_channels
            else None
        )
        self._3x3 = nn.Sequential(
            BasicConv2DBlock(in_channels, _3x3_reduction, kernel_size=1),
            BasicConv2DBlock(
                _3x3_reduction,
                _3x3_channels,
                kernel_size=3,
                stride=final_stride,
                padding=1,
            ),
        )
        self._5x5 = nn.Sequential(
            BasicConv2DBlock(in_channels, _5x5_reduction, kernel_size=1),
            BasicConv2DBlock(
                _5x5_reduction, _5x5_channels, kernel_size=3, padding=1
            ),
            BasicConv2DBlock(
                _5x5_channels,
                _5x5_channels,
                kernel_size=3,
                stride=final_stride,
                padding=1,
            ),
        )
        self.pooling = nn.Sequential(
            pooling_layer(
                3,
                stride=1 if pool_channels else final_stride,
                padding=1,
            ),
            BasicConv2DBlock(
                in_channels, pool_channels, kernel_size=1, stride=final_stride
            )
            if pool_channels
            else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = torch.cat((self._3x3(x), self._5x5(x), self.pooling(x)), 1)
        if self._1x1:
            out = torch.cat((self._1x1(x), out), 1)
        return out


class Inception_V2(nn.Module):
    """
    Inception-V2 model.

    Expects input of shape (3, 224, 224)
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        self.input = nn.Sequential(  # (3, 224, 224)
            BasicConv2DBlock(
                3, 64, kernel_size=7, stride=2, padding=3
            ),  # (64, 112, 112)
            nn.MaxPool2d(3, stride=2, padding=1),  # (64, 56, 56)
            BasicConv2DBlock(
                64, 192, kernel_size=3, padding=1
            ),  # (64, 56, 56)
            nn.MaxPool2d(3, stride=2, padding=1),  # (192, 28, 28)
        )

        self.inception_3a_to_3c = nn.Sequential(
            InceptionBlock(
                192, 64, 64, 64, 64, 96, InceptionBlock.AVG_POOL, 32
            ),  # (256, 28, 28)
            InceptionBlock(
                256, 64, 64, 96, 64, 96, InceptionBlock.AVG_POOL, 64
            ),  # (320, 28, 28)
            InceptionBlock(
                320, 0, 128, 160, 64, 96, InceptionBlock.MAX_POOL, None, 2
            ),  # (576, 14, 14)
        )

        self.inception_4a = InceptionBlock(
            576, 224, 64, 96, 96, 128, InceptionBlock.AVG_POOL, 128
        )  # (576, 14, 14)
        self.aux_1 = nn.Sequential(
            nn.AvgPool2d(5, stride=3),  # (576, 4, 4)
            BasicConv2DBlock(576, 128, kernel_size=1),  # (128, 4, 4),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.inception_4b_to_4d = nn.Sequential(
            InceptionBlock(
                576, 192, 96, 128, 96, 128, InceptionBlock.AVG_POOL, 128
            ),  # (576, 14, 14)
            InceptionBlock(
                576, 160, 128, 160, 128, 160, InceptionBlock.AVG_POOL, 128
            ),  # (608, 14, 14)
            InceptionBlock(
                608, 96, 128, 192, 160, 192, InceptionBlock.AVG_POOL, 128
            ),  # (608, 14, 14)
        )
        self.aux_2 = nn.Sequential(
            nn.AvgPool2d(5, stride=3),  # (608, 4, 4)
            BasicConv2DBlock(608, 128, kernel_size=1),  # (128, 4, 4),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.inception_4e = InceptionBlock(
            608, 0, 128, 192, 192, 256, InceptionBlock.MAX_POOL, None, 2
        )  # (1056, 7, 7)

        self.inception_5a_to_5b = nn.Sequential(
            InceptionBlock(
                1056, 352, 192, 320, 160, 224, InceptionBlock.AVG_POOL, 128
            ),  # (1024, 7, 7)
            InceptionBlock(
                1024, 352, 192, 320, 192, 224, InceptionBlock.MAX_POOL, 128
            ),  # (1024, 7, 7)
        )

        self.output = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.input(x)

        x = self.inception_3a_to_3c(x)

        x = self.inception_4a(x)
        if self.training:
            aux_1 = self.aux_1(x)
        x = self.inception_4b_to_4d(x)
        if self.training:
            aux_2 = self.aux_2(x)
        x = self.inception_4e(x)

        x = self.inception_5a_to_5b(x)

        x = self.output(x)
        if self.training:
            return x, aux_1, aux_2  # type: ignore
        return x
