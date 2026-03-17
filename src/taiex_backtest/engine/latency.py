"""Latency models for simulating order processing delays."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import numpy as np


class LatencyModel(ABC):
    """Abstract base for latency models."""

    @abstractmethod
    def get_delay(self) -> timedelta:
        """Return the simulated processing delay."""


class NoLatency(LatencyModel):
    """No latency - orders processed instantly."""

    def get_delay(self) -> timedelta:
        return timedelta(0)


class FixedLatency(LatencyModel):
    """Fixed latency delay in milliseconds."""

    def __init__(self, delay_ms: float = 50.0):
        if delay_ms < 0:
            raise ValueError(f"Delay must be non-negative: {delay_ms}")
        self._delay = timedelta(milliseconds=delay_ms)

    @property
    def delay_ms(self) -> float:
        return self._delay.total_seconds() * 1000

    def get_delay(self) -> timedelta:
        return self._delay


class RandomLatency(LatencyModel):
    """Random latency with configurable distribution.

    Uses log-normal distribution to simulate realistic network latency.
    """

    def __init__(
        self,
        mean_ms: float = 50.0,
        std_ms: float = 20.0,
        min_ms: float = 10.0,
        max_ms: float = 500.0,
        seed: int | None = None,
    ):
        if mean_ms < 0 or std_ms < 0:
            raise ValueError("Mean and std must be non-negative")
        self._mean = mean_ms
        self._std = std_ms
        self._min = min_ms
        self._max = max_ms
        self._rng = np.random.default_rng(seed)

    def get_delay(self) -> timedelta:
        delay_ms = self._rng.normal(self._mean, self._std)
        delay_ms = max(self._min, min(self._max, delay_ms))
        return timedelta(milliseconds=float(delay_ms))


def apply_latency(timestamp: datetime, latency: LatencyModel) -> datetime:
    """Apply latency to a timestamp."""
    return timestamp + latency.get_delay()
