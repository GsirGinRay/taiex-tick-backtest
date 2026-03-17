"""Walk-forward optimization and analysis."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

from ..analytics.equity_curve import get_equity_values
from ..analytics.metrics import PerformanceMetrics, calculate_metrics
from ..data.feed import DataFeed
from ..domain.enums import ProductType
from ..domain.models import Trade
from ..engine.backtest_engine import BacktestEngine
from ..strategy.base import Strategy
from .grid_search import grid_search


@dataclass(frozen=True)
class WalkForwardWindow:
    """Result for a single walk-forward window."""
    window_index: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    best_params: dict[str, Any]
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics
    out_of_sample_trades: list[Trade]


@dataclass(frozen=True)
class WalkForwardResult:
    """Complete walk-forward analysis result."""
    windows: list[WalkForwardWindow]
    combined_trades: list[Trade]
    combined_metrics: PerformanceMetrics
    n_windows: int
    in_sample_ratio: float
    objective: str


def _calculate_window_boundaries(
    start: datetime,
    end: datetime,
    n_windows: int,
    in_sample_ratio: float,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Calculate in-sample and out-of-sample boundaries for each window.
    
    Uses an anchored walk-forward approach where each window slides forward,
    maintaining the in_sample_ratio for the training period.
    
    Returns:
        List of (is_start, is_end, oos_start, oos_end) tuples.
    """
    total_duration = end - start
    # Each window's OOS covers 1/n_windows of the total OOS period
    # Total OOS fraction = 1 - in_sample_ratio of each window's total span
    
    # Simple sliding window approach:
    # Divide the data into n_windows + in_sample_windows overlapping segments
    window_step = total_duration / n_windows
    is_duration = total_duration * in_sample_ratio / 1.0
    
    # More practical: step size = total_duration / (n_windows + ratio * n_windows) ... 
    # Actually, let's use a simpler approach:
    # The OOS segments should cover the full period after the first IS window
    # Each OOS segment is window_step wide
    # IS for each window = preceding in_sample_ratio fraction of data up to OOS start
    
    oos_total = total_duration * (1 - in_sample_ratio)
    oos_per_window = oos_total / n_windows
    is_total = total_duration * in_sample_ratio
    
    boundaries = []
    for i in range(n_windows):
        oos_start = start + is_total + oos_per_window * i
        oos_end = oos_start + oos_per_window
        
        # In-sample: from start to oos_start (expanding window)
        # Or: fixed-size sliding window
        is_start = start + oos_per_window * i
        is_end = oos_start
        
        # Ensure boundaries don't exceed data range
        if oos_end > end:
            oos_end = end
        
        boundaries.append((is_start, is_end, oos_start, oos_end))
    
    return boundaries


def walk_forward(
    strategy_factory: Callable[..., Strategy],
    param_grid: dict[str, list],
    feed: DataFeed,
    n_windows: int = 5,
    in_sample_ratio: float = 0.7,
    objective: str = "sharpe_ratio",
    descending: bool = True,
    product: ProductType = ProductType.TX,
    initial_capital: Decimal = Decimal("1000000"),
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> WalkForwardResult:
    """Run walk-forward optimization.
    
    Splits data into rolling in-sample/out-of-sample windows.
    For each window:
        1. Run grid search on in-sample data to find best params
        2. Test best params on out-of-sample data
    
    Args:
        strategy_factory: Callable that creates Strategy from **kwargs.
        param_grid: Dict mapping param names to lists of values.
        feed: DataFeed with the full data period.
        n_windows: Number of walk-forward windows.
        in_sample_ratio: Fraction of each window used for in-sample (0.5-0.9).
        objective: Metric to optimize during in-sample grid search.
        descending: If True, higher objective is better.
        product: Product type.
        initial_capital: Starting capital.
        progress_callback: Optional callback(window_num, n_windows, phase).
    
    Returns:
        WalkForwardResult with per-window and combined results.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1: {n_windows}")
    if not (0.1 <= in_sample_ratio <= 0.95):
        raise ValueError(f"in_sample_ratio must be between 0.1 and 0.95: {in_sample_ratio}")
    
    start = feed.start_time
    end = feed.end_time
    
    boundaries = _calculate_window_boundaries(start, end, n_windows, in_sample_ratio)
    
    windows: list[WalkForwardWindow] = []
    all_oos_trades: list[Trade] = []
    running_capital = initial_capital
    
    for i, (is_start, is_end, oos_start, oos_end) in enumerate(boundaries):
        if progress_callback is not None:
            progress_callback(i + 1, n_windows, "in_sample")
        
        # In-sample: grid search
        is_feed = feed.slice(is_start, is_end)
        
        # Skip window if not enough data
        if is_feed.length < 10:
            continue
        
        is_results = grid_search(
            strategy_factory=strategy_factory,
            param_grid=param_grid,
            feed=is_feed,
            product=product,
            initial_capital=running_capital,
            objective=objective,
            descending=descending,
        )
        
        if not is_results:
            continue
        
        best = is_results[0]
        best_params = best.params
        is_metrics = best.metrics
        
        if progress_callback is not None:
            progress_callback(i + 1, n_windows, "out_of_sample")
        
        # Out-of-sample: test best params
        oos_feed = feed.slice(oos_start, oos_end)
        
        if oos_feed.length < 1:
            continue
        
        oos_strategy = strategy_factory(**best_params)
        oos_engine = BacktestEngine(
            feed=oos_feed,
            strategy=oos_strategy,
            product=product,
            initial_capital=running_capital,
        )
        oos_engine.run()
        
        oos_trades = oos_engine.trades
        oos_equity = get_equity_values(oos_trades, running_capital)
        oos_metrics = calculate_metrics(oos_trades, oos_equity, running_capital)
        
        # Update running capital for next window
        running_capital = oos_engine.capital
        
        all_oos_trades.extend(oos_trades)
        
        windows.append(WalkForwardWindow(
            window_index=i,
            in_sample_start=is_start,
            in_sample_end=is_end,
            out_of_sample_start=oos_start,
            out_of_sample_end=oos_end,
            best_params=best_params,
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            out_of_sample_trades=oos_trades,
        ))
    
    # Calculate combined metrics from all OOS trades
    combined_equity = get_equity_values(all_oos_trades, initial_capital)
    combined_metrics = calculate_metrics(all_oos_trades, combined_equity, initial_capital)
    
    return WalkForwardResult(
        windows=windows,
        combined_trades=all_oos_trades,
        combined_metrics=combined_metrics,
        n_windows=len(windows),
        in_sample_ratio=in_sample_ratio,
        objective=objective,
    )


def walk_forward_summary(result: WalkForwardResult) -> list[dict[str, Any]]:
    """Convert walk-forward result to summary dicts for display."""
    rows = []
    for w in result.windows:
        rows.append({
            "window": w.window_index + 1,
            "is_period": f"{w.in_sample_start:%Y-%m-%d} ~ {w.in_sample_end:%Y-%m-%d}",
            "oos_period": f"{w.out_of_sample_start:%Y-%m-%d} ~ {w.out_of_sample_end:%Y-%m-%d}",
            "best_params": w.best_params,
            "is_sharpe": f"{w.in_sample_metrics.sharpe_ratio:.2f}",
            "oos_sharpe": f"{w.out_of_sample_metrics.sharpe_ratio:.2f}",
            "is_net_pnl": str(w.in_sample_metrics.total_net_pnl),
            "oos_net_pnl": str(w.out_of_sample_metrics.total_net_pnl),
            "oos_trades": w.out_of_sample_metrics.total_trades,
        })
    return rows
