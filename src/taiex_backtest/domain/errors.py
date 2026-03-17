"""Domain-specific exceptions."""


class BacktestError(Exception):
    """Base exception for backtesting errors."""


class InsufficientMarginError(BacktestError):
    """Raised when margin is insufficient for an order."""


class InvalidOrderError(BacktestError):
    """Raised for invalid order parameters."""


class DataError(BacktestError):
    """Raised for data loading/processing errors."""


class StrategyError(BacktestError):
    """Raised for strategy-related errors."""


class ConfigError(BacktestError):
    """Raised for configuration errors."""
