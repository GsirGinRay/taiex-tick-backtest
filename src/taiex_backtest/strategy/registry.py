"""Strategy registry for dynamic strategy loading."""

from typing import Type

from .base import Strategy


class StrategyRegistry:
    """Registry for strategy classes."""

    def __init__(self):
        self._strategies: dict[str, Type[Strategy]] = {}

    def register(self, name: str | None = None):
        """Decorator to register a strategy class."""
        def decorator(cls: Type[Strategy]):
            key = name or cls.__name__
            self._strategies[key] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[Strategy]:
        """Get a strategy class by name."""
        if name not in self._strategies:
            raise KeyError(f"Strategy not found: {name}")
        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())

    def create(self, name: str, **kwargs) -> Strategy:
        """Create a strategy instance by name."""
        cls = self.get(name)
        return cls(**kwargs)


# Global registry instance
registry = StrategyRegistry()
