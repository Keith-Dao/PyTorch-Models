"""Dataset utils."""

import torch
import torch.utils


def get_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> torch.utils.data.DataLoader:
    """Creates a data loader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        batch_size: The batch size to use for the loader.

    Returns:
        A data loader for the given dataset.
    """
    assert hasattr(dataset, "__len__"), "Dataset must have a length."
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def sample_first(
    loader: torch.utils.data.DataLoader,
    classes: list[str],
) -> tuple[torch.Tensor, str]:
    """
    Sample the first item in the loader, returning the data and the associated
    label.

    Args:
        loader: The data loader to sample from.
        classes: The list of class names.

    Returns:
        A tuple containing the data and the associated label.
    """
    data, labels = next(iter(loader))
    return data[0].squeeze(0), classes[labels[0]]
