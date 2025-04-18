"""Metric utils."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
import torch


def plot_metric(histories: dict[str, dict[str, list[float]]], metric: str) -> None:
    """Plot the metrics."""
    ax = plt.figure().gca()

    for name, history in histories.items():
        plt.plot(
            range(1, len(history[metric]) + 1),
            history[metric],
            ".-",
            label=name.capitalize(),
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


def pretty_print_metrics(
    metrics: dict[str, list[torch.Tensor]], classes: list[str]
) -> None:
    """Print the metrics in a tabulated format."""
    single_value_headers, single_value_data = [], []
    multivalue_headers = ["Class"]
    multivalue_data: list[Any] = [classes]

    for metric, history in metrics.items():
        if history[-1].numel() == 1:
            headers = single_value_headers
            data = single_value_data
        else:
            headers = multivalue_headers
            data = multivalue_data
        headers.append(metric.capitalize())
        data.append(history[-1])

    float_format = ".4f"
    if single_value_headers:
        print(
            tabulate(
                [single_value_data],
                headers=single_value_headers,
                floatfmt=float_format,
            ),
            end="\n\n",
        )

    if len(multivalue_headers) > 1:
        multivalue_data = list(zip(*multivalue_data))
        print(
            tabulate(
                multivalue_data,
                headers=multivalue_headers,
                floatfmt=float_format,
            ),
            end="\n\n",
        )


def histories_to_md(
    classes: list[str],
    histories: list[dict[str, list[torch.Tensor]]],
    metrics: list[str],
) -> str:
    """Convert the metric histories into a markdown table."""
    table = list(
        zip(
            classes,
            *[
                [f"{x:.4f}" for x in history[metric][-1]]
                for metric in metrics
                for history in histories
            ],
        )
    )
    delimiter = "|"
    return "\n".join(f"{delimiter}{delimiter.join(row)}{delimiter}" for row in table)
