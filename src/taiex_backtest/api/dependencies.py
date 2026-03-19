"""FastAPI dependency injection."""

from pathlib import Path
from typing import Any

from ..strategy.registry import StrategyRegistry, registry as global_registry
from ..strategy.base import Strategy


class AppState:
    """Application state shared across requests."""

    def __init__(self):
        self._registry: StrategyRegistry = global_registry
        self._data_dir: Path = Path("data")
        self._running_tasks: dict[str, Any] = {}

    @property
    def registry(self) -> StrategyRegistry:
        return self._registry

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: Path) -> None:
        self._data_dir = value

    @property
    def running_tasks(self) -> dict[str, Any]:
        return self._running_tasks

    def register_strategy(
        self,
        name: str,
        strategy_class: type[Strategy],
    ) -> None:
        """Register a strategy class."""
        self._registry._strategies[name] = strategy_class

    def get_parquet_files(self) -> list[Path]:
        """List available parquet data files."""
        processed = self._data_dir / "processed"
        if not processed.exists():
            return []
        return sorted(processed.glob("*.parquet"))


# Global app state
app_state = AppState()


def get_app_state() -> AppState:
    """Dependency to get app state."""
    return app_state
