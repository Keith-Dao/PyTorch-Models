"""A manual scheduler."""


class ManualLRScheduler:
    learning_rates: list[float]
    counts: list[int]
    _i: int
    _count: int

    def __init__(self, learning_rates: list[float], counts: list[int]):
        """Scheduler to used along with the torch.optim.lr_scheduler.LambdaLR to
        change the learning rate to exact values after a certain number of
        epochs.

        Args:
            - learning_rates - The learning rate at each stage.
            - counts - The number of epochs before updating to the next stage
        """
        assert len(learning_rates) == len(counts) + 1, (
            "There should always be one more learning rate than count."
        )
        self.learning_rates = learning_rates
        self.counts = counts
        self._i = 0
        self._count = 0

    def step(self, _) -> float:
        """
        Step the scheduler to get the multiplicative factor for the learning
        rate.
        """
        if self._i < len(self.counts) and self._count == self.counts[self._i]:
            self._count = 0
            self._i += 1
        self._count += 1
        return self.learning_rates[self._i] / self.learning_rates[0]
