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
    ) -> None:
        super().__init__()

        self._1x1 = BasicConv2DBlock(in_channels, _1x1_channels, kernel_size=1)
        self._3x3 = nn.Sequential(
            BasicConv2DBlock(in_channels, _3x3_reduction, kernel_size=1),
            BasicConv2DBlock(
                _3x3_reduction, _3x3_channels, kernel_size=3, padding=1
            ),
        )
        self._5x5 = nn.Sequential(
            BasicConv2DBlock(in_channels, _5x5_reduction, kernel_size=1),
            BasicConv2DBlock(
                _5x5_reduction, _5x5_channels, kernel_size=3, padding=1
            ),
            BasicConv2DBlock(
                _5x5_reduction, _5x5_channels, kernel_size=3, padding=1
            ),
        )
        self.pooling = nn.Sequential(
            pooling_layer(3, stride=1, padding=1),
            BasicConv2DBlock(in_channels, pool_channels, kernel_size=1)
            if pool_channels
            else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.cat(
            (self._1x1(x), self._3x3(x), self._5x5(x), self.pooling(x)), 1
        )
