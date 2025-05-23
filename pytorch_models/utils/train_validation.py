"""Training and validation steps."""

from typing import Callable

import torch
import torch.utils
import torchmetrics
import tqdm.auto

from .metrics import pretty_print_metrics


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_classes: int,
    metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
    tqdm_description: str = "",
    auxiliary_loss_weight: float = 0.3,
) -> torch.Tensor:
    """Trains the model for one epoch.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        scheduler: The learning rate scheduler to use.
        loader: The data loader to use.
        loss_fn: The loss function to use.
        num_classes: The number of classes in the dataset.
        metrics: The metrics to compute.
        device: The device to use.
        tqdm_description: Description for the tqdm progress bar.
        auxiliary_loss_weight: Weight for the auxiliary loss.

    Returns:
        The average training loss for the epoch.
    """
    training_loss = torch.tensor([0], dtype=torch.float, device=device)
    count = 0
    for data, targets in tqdm.tqdm(loader, desc=tqdm_description, ncols=100):
        count += len(data)
        data = data.to(device)
        targets = targets.to(device)
        y = torch.nn.functional.one_hot(targets, num_classes).float()

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(data)
        if isinstance(y_pred, tuple):
            y_pred, *aux_preds = y_pred
            loss = loss_fn(y_pred, y) + auxiliary_loss_weight * sum(
                loss_fn(aux_pred, y) for aux_pred in aux_preds
            )
        else:
            loss = loss_fn(y_pred, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        training_loss += loss.detach()
        if metrics:
            metrics.update(y_pred, targets)
    if scheduler:
        scheduler.step()
    return training_loss.to("cpu") / count


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_classes: int,
    metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
    tqdm_description: str = "",
) -> torch.Tensor:
    """Validates the model for one epoch.

    Args:
        model: The model to validate.
        loader: The data loader to use.
        loss_fn: The loss function to use.
        num_classes: The number of classes in the dataset.
        metrics: The metrics to compute.
        device: The device to use.
        tqdm_description: Description for the tqdm progress bar.

    Returns:
        The average validation loss for the epoch.
    """
    model.eval()
    validation_loss = torch.tensor([0], dtype=torch.float, device=device)
    count = 0
    for data, targets in tqdm.tqdm(loader, desc=tqdm_description, ncols=100):
        count += len(data)
        data = data.to(device)
        targets = targets.to(device)
        y = torch.nn.functional.one_hot(targets, num_classes).float()

        # Forward pass
        y_pred = model(data)
        loss = loss_fn(y_pred, y)

        # Metrics
        validation_loss += loss
        if metrics:
            metrics.update(y_pred, targets)
    model.train()
    return validation_loss.to("cpu") / count


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_loader: torch.utils.data.DataLoader,
    train_history: dict[str, list[torch.Tensor]],
    validation_loader: torch.utils.data.DataLoader,
    validation_history: dict[str, list[torch.Tensor]],
    epochs: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    classes: list[str],
    train_metrics: torchmetrics.MetricCollection | None = None,
    validation_metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
    auxiliary_loss_weight: float = 0.3,
):
    """Trains and validates the model for the given number of epochs.

    Args:
        model: The model to train.
        optimizer: The optimizer to use.
        scheduler: The learning rate scheduler to use.
        train_loader: The training data loader.
        train_history: The training history to update.
        validation_loader: The validation data loader.
        validation_history: The validation history to update.
        epochs: The number of epochs to train for.
        loss_fn: The loss function to use.
        classes: The list of class names.
        train_metrics: The metrics to compute during training.
        validation_metrics: The metrics to compute during validation.
        device: The device to use.
        auxiliary_loss_weight: Weight for the auxiliary loss.
    """
    for epoch in range(1, epochs + 1):
        training_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            train_loader,
            loss_fn,
            len(classes),
            train_metrics,
            device,
            f"Training epoch {epoch}/{epochs}",
            auxiliary_loss_weight,
        )
        if train_metrics:
            for metric, value in train_metrics.compute().items():
                train_history[metric].append(value.to("cpu"))
            train_metrics.reset()
        train_history["loss"].append(training_loss)
        pretty_print_metrics(train_history, classes)

        validation_loss = validate_one_epoch(
            model,
            validation_loader,
            loss_fn,
            len(classes),
            validation_metrics,
            device,
            f"Validating epoch {epoch}/{epochs}",
        )
        if validation_metrics:
            for metric, value in validation_metrics.compute().items():
                validation_history[metric].append(value.to("cpu"))
            validation_metrics.reset()
        validation_history["loss"].append(validation_loss)
        pretty_print_metrics(validation_history, classes)
