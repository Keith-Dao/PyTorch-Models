"""Data augmentations."""

import torch


class AddGaussianNoise:
    mean: float
    std: float

    def __init__(self, mean: float = 0, std: float = 1):
        """Augment that randomly adds gaussian noise.

        Args:
            mean: Mean of the gaussian noise.
            std: Standard deviation of the gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, input_: torch.Tensor) -> torch.Tensor:
        return input_ + torch.randn(input_.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (mean={self.mean}, std={self.std})"


class Clip:
    min_: float | None
    max_: float | None

    def __init__(self, min_: float | None = None, max_: float | None = None):
        """Augment that clips the input to the given range.

        Args:
            min_: Minimum value to clip to. If None, no minimum clipping is
                applied.
            max_: Maximum value to clip to. If None, no maximum clipping is
                applied.
        """
        if min_ is not None and max_ is not None and min_ >= max_:
            raise ValueError("min_ must be strictly less than max_.")
        self.min = min_
        self.max = max_

    def __call__(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input_, self.min, self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (min={self.min}, max={self.max})"
