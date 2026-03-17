"""Signal helpers for strategy development."""

from collections import deque
from decimal import Decimal


class MovingAverage:
    """Simple moving average calculator for streaming data."""

    def __init__(self, period: int):
        if period <= 0:
            raise ValueError(f"Period must be positive: {period}")
        self._period = period
        self._values: deque[Decimal] = deque(maxlen=period)
        self._sum = Decimal("0")

    @property
    def period(self) -> int:
        return self._period

    @property
    def is_ready(self) -> bool:
        return len(self._values) == self._period

    @property
    def value(self) -> Decimal | None:
        if not self.is_ready:
            return None
        return self._sum / self._period

    def update(self, price: Decimal) -> Decimal | None:
        """Update with a new price and return the current MA value."""
        if len(self._values) == self._period:
            self._sum -= self._values[0]
        self._values.append(price)
        self._sum += price
        return self.value

    def reset(self) -> None:
        self._values.clear()
        self._sum = Decimal("0")


class ExponentialMovingAverage:
    """Exponential moving average calculator for streaming data."""

    def __init__(self, period: int):
        if period <= 0:
            raise ValueError(f"Period must be positive: {period}")
        self._period = period
        self._multiplier = Decimal("2") / (Decimal(str(period)) + Decimal("1"))
        self._value: Decimal | None = None
        self._count = 0
        self._sum = Decimal("0")

    @property
    def period(self) -> int:
        return self._period

    @property
    def is_ready(self) -> bool:
        return self._count >= self._period

    @property
    def value(self) -> Decimal | None:
        return self._value

    def update(self, price: Decimal) -> Decimal | None:
        """Update with a new price and return the current EMA value."""
        self._count += 1

        if self._count <= self._period:
            self._sum += price
            if self._count == self._period:
                self._value = self._sum / self._period
            return self._value

        if self._value is not None:
            self._value = (price - self._value) * self._multiplier + self._value

        return self._value

    def reset(self) -> None:
        self._value = None
        self._count = 0
        self._sum = Decimal("0")


class CrossDetector:
    """Detects crossover and crossunder events between two values."""

    def __init__(self):
        self._prev_a: Decimal | None = None
        self._prev_b: Decimal | None = None

    def update(self, a: Decimal, b: Decimal) -> tuple[bool, bool]:
        """Update with new values. Returns (crossover, crossunder).

        crossover: a crosses above b
        crossunder: a crosses below b
        """
        crossover = False
        crossunder = False

        if self._prev_a is not None and self._prev_b is not None:
            if self._prev_a <= self._prev_b and a > b:
                crossover = True
            elif self._prev_a >= self._prev_b and a < b:
                crossunder = True

        self._prev_a = a
        self._prev_b = b
        return crossover, crossunder

    def reset(self) -> None:
        self._prev_a = None
        self._prev_b = None
