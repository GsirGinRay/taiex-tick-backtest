"""Simulation clock for the backtesting engine."""

from datetime import datetime, time
from typing import Optional

from ..domain.enums import Session


# Trading session time boundaries
DAY_SESSION_START = time(8, 45)
DAY_SESSION_END = time(13, 45)
NIGHT_SESSION_START = time(15, 0)
NIGHT_SESSION_END = time(5, 0)  # Next day


class Clock:
    """Tracks the current simulation time."""

    def __init__(self):
        self._current_time: Optional[datetime] = None
        self._tick_count: int = 0

    @property
    def now(self) -> Optional[datetime]:
        return self._current_time

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def update(self, timestamp: datetime) -> None:
        """Advance the clock to a new timestamp."""
        if self._current_time is not None and timestamp < self._current_time:
            raise ValueError(
                f"Clock cannot go backwards: {timestamp} < {self._current_time}"
            )
        self._current_time = timestamp
        self._tick_count += 1

    def reset(self) -> None:
        """Reset the clock to initial state."""
        self._current_time = None
        self._tick_count = 0

    @staticmethod
    def get_session(timestamp: datetime) -> Session:
        """Determine the trading session for a given timestamp."""
        t = timestamp.time()
        if DAY_SESSION_START <= t <= DAY_SESSION_END:
            return Session.DAY
        return Session.NIGHT

    @staticmethod
    def is_trading_hours(timestamp: datetime) -> bool:
        """Check if a timestamp falls within trading hours."""
        t = timestamp.time()
        # Day session: 08:45 - 13:45
        if DAY_SESSION_START <= t <= DAY_SESSION_END:
            return True
        # Night session: 15:00 - 23:59 or 00:00 - 05:00
        if t >= NIGHT_SESSION_START:
            return True
        if t <= NIGHT_SESSION_END:
            return True
        return False
