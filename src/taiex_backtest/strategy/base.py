"""Base strategy abstract class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..domain.models import Tick

if TYPE_CHECKING:
    from .context import StrategyContext


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    def name(self) -> str:
        """Strategy name, defaults to class name."""
        return self.__class__.__name__

    def on_init(self, ctx: "StrategyContext") -> None:
        """Called once when the backtest starts. Override to initialize."""

    @abstractmethod
    def on_tick(self, ctx: "StrategyContext", tick: Tick) -> None:
        """Called on every tick. Must be implemented by subclasses."""

    def on_stop(self, ctx: "StrategyContext") -> None:
        """Called once when the backtest ends. Override for cleanup."""
