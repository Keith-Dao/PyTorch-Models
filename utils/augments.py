"""
    Data augmentations.
"""
import torch


class AddGaussianNoise:
    """
    Randomly adds gaussian noise.
    """

    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, input_: torch.Tensor) -> torch.Tensor:
        return input_ + torch.randn(input_.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (mean={self.mean}, std={self.std})"


class Clip:
    """
    Clips input to the given range.
    """

    def __init__(
        self, min_: float | None = None, max_: float | None = None
    ) -> None:
        if min_ is not None and max_ is not None and min_ >= max_:
            raise ValueError("min_ must be strictly less than max_.")
        self.min = min_
        self.max = max_

    def __call__(self, input_: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input_, self.min, self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} (min={self.min}, max={self.max})"
