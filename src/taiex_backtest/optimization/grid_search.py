"""Grid search parameter optimization."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable
import itertools

from ..analytics.equity_curve import get_equity_values
from ..analytics.metrics import PerformanceMetrics, calculate_metrics
from ..data.feed import DataFeed
from ..domain.enums import ProductType
from ..domain.models import Trade
from ..engine.backtest_engine import BacktestEngine
from ..strategy.base import Strategy


@dataclass(frozen=True)
class GridSearchResult:
    """Result of a single parameter combination backtest."""
    params: dict[str, Any]
    metrics: PerformanceMetrics
    trades: list[Trade]
    objective_value: float


def _run_single_backtest(
    strategy_factory: Callable[..., Strategy],
    params: dict[str, Any],
    feed: DataFeed,
    product: ProductType,
    initial_capital: Decimal,
    objective: str,
) -> GridSearchResult:
    """Run a single backtest with given params and return result."""
    strategy = strategy_factory(**params)
    engine = BacktestEngine(
        feed=feed,
        strategy=strategy,
        product=product,
        initial_capital=initial_capital,
    )
    engine.run()
    trades = engine.trades
    equity_values = get_equity_values(trades, initial_capital)
    metrics = calculate_metrics(trades, equity_values, initial_capital)
    
    obj_value = getattr(metrics, objective)
    if isinstance(obj_value, Decimal):
        obj_value = float(obj_value)
    
    return GridSearchResult(
        params=params,
        metrics=metrics,
        trades=trades,
        objective_value=obj_value,
    )


def grid_search(
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list],
    feed: DataFeed,
    product: ProductType = ProductType.TX,
    initial_capital: Decimal = Decimal("1000000"),
    objective: str = "sharpe_ratio",
    descending: bool = True,
    n_jobs: int = 1,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> list[GridSearchResult]:
    """Run grid search over parameter combinations.
    
    Args:
        strategy_factory: Callable that creates Strategy from **kwargs.
        param_grid: Dict mapping param names to lists of values.
            Example: {"fast_period": [10, 20, 30], "slow_period": [50, 100]}
        feed: DataFeed to use for all backtests.
        product: Product type.
        initial_capital: Starting capital.
        objective: Metric attribute name to optimize.
        descending: If True, higher objective is better.
        n_jobs: Number of parallel workers (1=sequential).
        progress_callback: Optional callback(current, total, params).
    
    Returns:
        List of GridSearchResult sorted by objective_value.
    """
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    total = len(combinations)
    if total == 0:
        return []
    
    results: list[GridSearchResult] = []
    
    if n_jobs <= 1:
        # Sequential execution
        for i, params in enumerate(combinations):
            if progress_callback is not None:
                progress_callback(i + 1, total, params)
            result = _run_single_backtest(
                strategy_factory, params, feed, product, initial_capital, objective
            )
            results.append(result)
    else:
        # Parallel execution
        from .parallel import parallel_map
        
        def task(params):
            return _run_single_backtest(
                strategy_factory, params, feed, product, initial_capital, objective
            )
        
        results = parallel_map(task, combinations, n_jobs=n_jobs)
    
    # Sort by objective
    results.sort(key=lambda r: r.objective_value, reverse=descending)
    return results


def grid_search_summary(results: list[GridSearchResult]) -> list[dict[str, Any]]:
    """Convert grid search results to a list of summary dicts for display.
    
    Returns list of dicts with params + key metrics, sorted by objective.
    """
    rows = []
    for i, r in enumerate(results, 1):
        row: dict[str, Any] = {"rank": i}
        row.update(r.params)
        row["total_trades"] = r.metrics.total_trades
        row["win_rate"] = f"{r.metrics.win_rate:.1%}"
        row["net_pnl"] = str(r.metrics.total_net_pnl)
        row["sharpe"] = f"{r.metrics.sharpe_ratio:.2f}"
        row["max_dd_pct"] = f"{r.metrics.max_drawdown_pct:.2%}"
        row["profit_factor"] = f"{r.metrics.profit_factor:.2f}"
        row["objective"] = r.objective_value
        rows.append(row)
    return rows
