"""Multi-strategy comparison utilities."""

from dataclasses import dataclass
from decimal import Decimal

from .metrics import PerformanceMetrics


@dataclass(frozen=True)
class StrategyComparison:
    """Side-by-side comparison of multiple strategies."""
    names: list[str]
    metrics: list[PerformanceMetrics]

    @property
    def count(self) -> int:
        return len(self.names)


def compare_strategies(
    results: dict[str, PerformanceMetrics],
) -> StrategyComparison:
    """Compare multiple strategy results side-by-side.

    Args:
        results: Dict mapping strategy name to its PerformanceMetrics.

    Returns:
        StrategyComparison with all strategies.
    """
    names = list(results.keys())
    metrics = list(results.values())
    return StrategyComparison(names=names, metrics=metrics)


def rank_by(
    comparison: StrategyComparison,
    metric: str,
    descending: bool = True,
) -> list[tuple[str, float | Decimal]]:
    """Rank strategies by a specific metric.

    Args:
        comparison: StrategyComparison object.
        metric: Attribute name on PerformanceMetrics to rank by.
        descending: If True, higher is better (e.g., Sharpe, PnL).

    Returns:
        List of (strategy_name, metric_value) sorted by rank.
    """
    pairs: list[tuple[str, float | Decimal]] = []
    for name, m in zip(comparison.names, comparison.metrics):
        value = getattr(m, metric)
        pairs.append((name, value))

    pairs.sort(key=lambda x: float(x[1]) if isinstance(x[1], Decimal) else x[1],
               reverse=descending)
    return pairs


def comparison_to_table(comparison: StrategyComparison) -> list[dict]:
    """Convert comparison to a list of dicts for tabular display.

    Each dict has 'metric' key and one key per strategy name.
    """
    key_metrics = [
        ("total_trades", "Total Trades"),
        ("win_rate", "Win Rate"),
        ("total_net_pnl", "Net PnL"),
        ("profit_factor", "Profit Factor"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("sortino_ratio", "Sortino Ratio"),
        ("max_drawdown", "Max Drawdown"),
        ("max_drawdown_pct", "Max DD %"),
        ("calmar_ratio", "Calmar Ratio"),
        ("expectancy", "Expectancy"),
        ("payoff_ratio", "Payoff Ratio"),
        ("max_consecutive_wins", "Max Consecutive Wins"),
        ("max_consecutive_losses", "Max Consecutive Losses"),
    ]

    rows: list[dict] = []
    for attr, label in key_metrics:
        row: dict = {"metric": label}
        for name, m in zip(comparison.names, comparison.metrics):
            value = getattr(m, attr)
            if isinstance(value, Decimal):
                row[name] = str(value)
            elif isinstance(value, float):
                row[name] = f"{value:.4f}"
            else:
                row[name] = str(value)
        rows.append(row)

    return rows
