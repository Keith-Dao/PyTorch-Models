"""
    Dataset utils.
"""

import torch
import torch.utils


def get_loader(
    dataset: torch.utils.data.Dataset, batch_size: int
) -> torch.utils.data.DataLoader:
    """
    Create a data loader for the given dataset.
    """
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )


def sample_first(
    loader: torch.utils.data.DataLoader, classes: list[str]
) -> tuple[torch.Tensor, str]:
    """
    Sample the first item in the loader, returning the data and the associated
    label.
    """
    data, labels = next(iter(loader))
    return data[0].squeeze(0), classes[labels[0]]
