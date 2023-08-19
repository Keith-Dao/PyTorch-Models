"""
    Training and validation steps.
"""
from typing import Callable

import torch
import torchmetrics
import tqdm.auto

from .metrics import pretty_print_metrics


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
    num_classes: int,
    metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
    tqdm_description: str = "",
) -> torch.tensor:
    """
    One training epoch.
    """
    training_loss = torch.tensor([0], dtype=torch.float, device=device)
    for data, targets in tqdm.tqdm(loader, desc=tqdm_description, ncols=100):
        data = data.to(device)
        targets = targets.to(device)
        y = torch.nn.functional.one_hot(targets, num_classes).float()

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(data)
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
    return training_loss.to("cpu") / len(loader.dataset)


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
    num_classes: int,
    metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
    tqdm_description: str = "",
) -> torch.tensor:
    """
    One validation epoch.
    """
    validation_loss = torch.tensor([0], dtype=torch.float, device=device)
    for data, targets in tqdm.tqdm(loader, desc=tqdm_description, ncols=100):
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
    return validation_loss.to("cpu") / len(loader.dataset)


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_loader: torch.utils.data.DataLoader,
    train_history: dict[str, list[float]],
    validation_loader: torch.utils.data.DataLoader,
    validation_history: dict[str, list[float]],
    epochs: int,
    loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
    classes: list[str],
    train_metrics: torchmetrics.MetricCollection | None = None,
    validation_metrics: torchmetrics.MetricCollection | None = None,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Train and validation the model for the given number of epochs.
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
