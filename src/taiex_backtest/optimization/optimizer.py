"""Optuna-based Bayesian parameter optimization."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable

from ..analytics.equity_curve import get_equity_values
from ..analytics.metrics import PerformanceMetrics, calculate_metrics
from ..data.feed import DataFeed
from ..domain.enums import ProductType
from ..domain.models import Trade
from ..engine.backtest_engine import BacktestEngine
from ..strategy.base import Strategy


@dataclass(frozen=True)
class OptimizationResult:
    """Result of an Optuna optimization run."""
    best_params: dict[str, Any]
    best_value: float
    best_metrics: PerformanceMetrics
    best_trades: list[Trade]
    all_trials: list[dict[str, Any]]
    n_trials: int
    objective_name: str
    direction: str


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single parameter to optimize.
    
    Attributes:
        name: Parameter name (kwarg to strategy factory).
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        step: Step size. If None, continuous float range.
        param_type: "int" for integer, "float" for continuous.
        log: If True, sample in log scale (useful for learning rates).
    """
    name: str
    low: float
    high: float
    step: float | None = None
    param_type: str = "int"
    log: bool = False


def _suggest_param(trial, spec: ParamSpec) -> int | float:
    """Suggest a parameter value from Optuna trial."""
    if spec.param_type == "int":
        step = int(spec.step) if spec.step is not None else 1
        return trial.suggest_int(
            spec.name, int(spec.low), int(spec.high), step=step, log=spec.log,
        )
    else:
        return trial.suggest_float(
            spec.name, spec.low, spec.high,
            step=spec.step, log=spec.log,
        )


def optimize(
    strategy_factory: Callable[..., Strategy],
    param_specs: list[ParamSpec],
    feed: DataFeed,
    n_trials: int = 100,
    objective: str = "sharpe_ratio",
    direction: str = "maximize",
    product: ProductType = ProductType.TX,
    initial_capital: Decimal = Decimal("1000000"),
    seed: int | None = 42,
    timeout: float | None = None,
    progress_callback: Callable[[int, int, dict, float], None] | None = None,
) -> OptimizationResult:
    """Run Optuna Bayesian optimization to find best strategy parameters.
    
    Args:
        strategy_factory: Callable that creates Strategy from **kwargs.
        param_specs: List of ParamSpec defining the search space.
        feed: DataFeed to use for all backtests.
        n_trials: Number of optimization trials.
        objective: Metric attribute name to optimize.
        direction: "maximize" or "minimize".
        product: Product type.
        initial_capital: Starting capital.
        seed: Random seed for reproducibility.
        timeout: Optional timeout in seconds.
        progress_callback: Optional callback(trial_num, n_trials, params, value).
    
    Returns:
        OptimizationResult with best parameters and all trial info.
    
    Raises:
        ImportError: If optuna is not installed.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for Bayesian optimization. "
            "Install it with: pip install taiex-backtest[optimization]"
        )
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    all_trials: list[dict[str, Any]] = []
    best_metrics: PerformanceMetrics | None = None
    best_trades: list[Trade] = []
    best_value: float = float("-inf") if direction == "maximize" else float("inf")
    best_params: dict[str, Any] = {}
    trial_count = 0
    
    def objective_fn(trial):
        nonlocal best_metrics, best_trades, best_value, best_params, trial_count
        
        # Suggest parameters
        params: dict[str, Any] = {}
        for spec in param_specs:
            params[spec.name] = _suggest_param(trial, spec)
        
        # Run backtest
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
        
        trial_count += 1
        
        # Track all trials
        trial_info = {
            "number": trial_count,
            "params": dict(params),
            "value": obj_value,
            "total_trades": metrics.total_trades,
            "net_pnl": str(metrics.total_net_pnl),
            "sharpe_ratio": metrics.sharpe_ratio,
            "win_rate": metrics.win_rate,
        }
        all_trials.append(trial_info)
        
        # Track best
        is_better = (
            (direction == "maximize" and obj_value > best_value)
            or (direction == "minimize" and obj_value < best_value)
        )
        if is_better:
            best_value = obj_value
            best_params = dict(params)
            best_metrics = metrics
            best_trades = trades
        
        if progress_callback is not None:
            progress_callback(trial_count, n_trials, params, obj_value)
        
        return obj_value
    
    # Create study
    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)
    
    # If no trials ran successfully
    if best_metrics is None:
        from ..analytics.metrics import _empty_metrics
        best_metrics = _empty_metrics()
    
    return OptimizationResult(
        best_params=best_params,
        best_value=best_value,
        best_metrics=best_metrics,
        best_trades=best_trades,
        all_trials=all_trials,
        n_trials=trial_count,
        objective_name=objective,
        direction=direction,
    )
