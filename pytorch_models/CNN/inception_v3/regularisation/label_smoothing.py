"""
Model regularisation via label smoothing.
"""

import torch
from torch import nn


class LabelSmoothing(nn.Module):
    """Label smoothing."""

    def __init__(self, loss: nn.Module, epsilon: float) -> None:
        super().__init__()
        if not 0 <= epsilon < 1:
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        self.loss = loss
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to loss function."""
        num_classes = input.shape[1]
        return (1 - self.epsilon) * self.loss(input, target) + self.epsilon * self.loss(
            input, torch.full_like(input, 1 / num_classes)
        )
