"""
    Dataset utils.
"""

import torch


def sample_first(
    loader: torch.utils.data.DataLoader, classes: list[str]
) -> tuple[torch.Tensor, str]:
    """
    Sample the first item in the loader, returning the data and the associated
    label.
    """
    data, labels = next(iter(loader))
    return data[0].squeeze(0), classes[labels[0]]
