"""
    Visualisation utils.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_metric(histories: dict[str, dict[str, list[float]]], metric: str):
    """
    Plot the metrics.
    """
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
