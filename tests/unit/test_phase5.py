"""Tests for Phase 5: optimization modules.

Covers:
- parallel.py: get_max_workers, parallel_map, chunked
- grid_search.py: GridSearchResult, _run_single_backtest, grid_search, grid_search_summary
- optimizer.py: ParamSpec, OptimizationResult, optimize (requires optuna)
- walk_forward.py: WalkForwardWindow, WalkForwardResult, walk_forward, walk_forward_summary
"""

from decimal import Decimal

import pytest

from taiex_backtest.analytics.metrics import PerformanceMetrics, _empty_metrics
from taiex_backtest.data.feed import DataFeed
from taiex_backtest.data.synthetic import generate_ticks
from taiex_backtest.data.writer import ticks_to_dataframe
from taiex_backtest.domain.enums import ProductType
from taiex_backtest.optimization.grid_search import (
    GridSearchResult,
    _run_single_backtest,
    grid_search,
    grid_search_summary,
)
from taiex_backtest.optimization.parallel import chunked, get_max_workers, parallel_map
from taiex_backtest.strategy.base import Strategy


# ---------------------------------------------------------------------------
# Helper strategy
# ---------------------------------------------------------------------------


class SimpleParamStrategy(Strategy):
    """Parameterized strategy for testing optimization.

    Buys after ``warmup`` ticks, holds for ``hold_period`` ticks, then sells.
    Repeats until the data feed is exhausted.
    """

    def __init__(
        self,
        warmup: int = 10,
        hold_period: int = 5,
        quantity: int = 1,
    ) -> None:
        self._warmup = warmup
        self._hold_period = hold_period
        self._quantity = quantity
        self._tick_count = 0
        self._holding_for = 0
        self._in_position = False

    def on_tick(self, ctx, tick) -> None:
        self._tick_count += 1
        if self._in_position:
            self._holding_for += 1
            if self._holding_for >= self._hold_period:
                ctx.close_position(tag="param_exit")
                self._in_position = False
                self._holding_for = 0
        elif self._tick_count >= self._warmup and ctx.position.is_flat:
            ctx.buy(quantity=self._quantity, tag="param_entry")
            self._in_position = True
            self._holding_for = 0

    def on_stop(self, ctx) -> None:
        if not ctx.position.is_flat:
            ctx.close_position(tag="param_stop")


def make_strategy(**kwargs) -> SimpleParamStrategy:
    """Factory function for SimpleParamStrategy."""
    return SimpleParamStrategy(**kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_feed() -> DataFeed:
    """Create a small DataFeed (500 ticks) for fast tests."""
    ticks = generate_ticks(start_price=20000.0, num_ticks=500, seed=42)
    df = ticks_to_dataframe(ticks)
    return DataFeed(df)


@pytest.fixture
def medium_feed() -> DataFeed:
    """Create a medium DataFeed (2000 ticks) for walk-forward tests."""
    ticks = generate_ticks(start_price=20000.0, num_ticks=2000, seed=42)
    df = ticks_to_dataframe(ticks)
    return DataFeed(df)


# ===================================================================
# parallel.py
# ===================================================================


class TestGetMaxWorkers:
    """Tests for get_max_workers()."""

    def test_positive_number(self) -> None:
        assert get_max_workers(4) == 4

    def test_one_returns_one(self) -> None:
        assert get_max_workers(1) == 1

    def test_negative_one_returns_cpu_count(self) -> None:
        import os

        expected = os.cpu_count() or 1
        assert get_max_workers(-1) == expected

    def test_zero_returns_one(self) -> None:
        assert get_max_workers(0) == 1


class TestParallelMap:
    """Tests for parallel_map()."""

    def test_empty_list(self) -> None:
        result = parallel_map(lambda x: x * 2, [])
        assert result == []

    def test_sequential_single_worker(self) -> None:
        result = parallel_map(lambda x: x * 2, [1, 2, 3], n_jobs=1)
        assert result == [2, 4, 6]

    def test_parallel_preserves_order(self) -> None:
        result = parallel_map(lambda x: x ** 2, list(range(20)), n_jobs=2)
        assert result == [x ** 2 for x in range(20)]

    def test_with_progress_callback(self) -> None:
        progress: list[tuple[int, int]] = []

        def callback(done: int, total: int) -> None:
            progress.append((done, total))

        result = parallel_map(lambda x: x, [1, 2, 3], n_jobs=1, progress_callback=callback)
        assert result == [1, 2, 3]
        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_parallel_multiple_workers(self) -> None:
        result = parallel_map(lambda x: x + 1, list(range(10)), n_jobs=3)
        assert result == list(range(1, 11))

    def test_large_batch_parallel(self) -> None:
        items = list(range(100))
        result = parallel_map(lambda x: x * 3, items, n_jobs=4)
        assert result == [x * 3 for x in items]


class TestChunked:
    """Tests for chunked()."""

    def test_even_split(self) -> None:
        result = chunked([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self) -> None:
        result = chunked([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_chunk_size_larger_than_list(self) -> None:
        result = chunked([1, 2], 10)
        assert result == [[1, 2]]

    def test_empty_list(self) -> None:
        result = chunked([], 5)
        assert result == []

    def test_chunk_size_one(self) -> None:
        result = chunked([1, 2, 3], 1)
        assert result == [[1], [2], [3]]

    def test_invalid_chunk_size_zero(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            chunked([1], 0)

    def test_invalid_chunk_size_negative(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            chunked([1], -1)

    def test_single_element(self) -> None:
        result = chunked([42], 3)
        assert result == [[42]]


# ===================================================================
# grid_search.py
# ===================================================================


class TestGridSearchResult:
    """Tests for GridSearchResult dataclass."""

    def test_fields(self) -> None:
        metrics = _empty_metrics()
        r = GridSearchResult(
            params={"a": 1},
            metrics=metrics,
            trades=[],
            objective_value=0.5,
        )
        assert r.params == {"a": 1}
        assert r.metrics is metrics
        assert r.trades == []
        assert r.objective_value == 0.5

    def test_frozen(self) -> None:
        r = GridSearchResult(
            params={"a": 1},
            metrics=_empty_metrics(),
            trades=[],
            objective_value=0.5,
        )
        with pytest.raises(AttributeError):
            r.objective_value = 1.0  # type: ignore[misc]


class TestRunSingleBacktest:
    """Tests for _run_single_backtest()."""

    def test_returns_grid_search_result(self, small_feed: DataFeed) -> None:
        result = _run_single_backtest(
            strategy_factory=make_strategy,
            params={"warmup": 10, "hold_period": 5},
            feed=small_feed,
            product=ProductType.TX,
            initial_capital=Decimal("1000000"),
            objective="sharpe_ratio",
        )
        assert isinstance(result, GridSearchResult)
        assert result.params == {"warmup": 10, "hold_period": 5}
        assert isinstance(result.objective_value, float)

    def test_metrics_populated(self, small_feed: DataFeed) -> None:
        result = _run_single_backtest(
            strategy_factory=make_strategy,
            params={"warmup": 10, "hold_period": 5},
            feed=small_feed,
            product=ProductType.TX,
            initial_capital=Decimal("1000000"),
            objective="total_net_pnl",
        )
        assert isinstance(result.metrics, PerformanceMetrics)

    def test_different_objectives(self, small_feed: DataFeed) -> None:
        for obj in ("sharpe_ratio", "total_net_pnl", "win_rate", "profit_factor"):
            result = _run_single_backtest(
                strategy_factory=make_strategy,
                params={"warmup": 10, "hold_period": 5},
                feed=small_feed,
                product=ProductType.TX,
                initial_capital=Decimal("1000000"),
                objective=obj,
            )
            assert isinstance(result.objective_value, float)


class TestGridSearch:
    """Tests for grid_search()."""

    def test_single_combination(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10], "hold_period": [5]},
            feed=small_feed,
        )
        assert len(results) == 1
        assert results[0].params == {"warmup": 10, "hold_period": 5}

    def test_multiple_combinations(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20], "hold_period": [5, 10]},
            feed=small_feed,
        )
        assert len(results) == 4  # 2 x 2

    def test_sorted_by_objective_descending(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 50, 100], "hold_period": [5, 20]},
            feed=small_feed,
            objective="sharpe_ratio",
            descending=True,
        )
        values = [r.objective_value for r in results]
        assert values == sorted(values, reverse=True)

    def test_sorted_ascending(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 50], "hold_period": [5, 10]},
            feed=small_feed,
            objective="max_drawdown_pct",
            descending=False,
        )
        values = [r.objective_value for r in results]
        assert values == sorted(values)

    def test_empty_param_grid(self, small_feed: DataFeed) -> None:
        # An empty param_grid still produces one combination (default params)
        # because itertools.product(*[]) yields one empty tuple.
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={},
            feed=small_feed,
        )
        assert len(results) == 1
        assert results[0].params == {}

    def test_progress_callback(self, small_feed: DataFeed) -> None:
        progress: list[tuple[int, int]] = []

        def callback(current: int, total: int, params: dict) -> None:
            progress.append((current, total))

        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20], "hold_period": [5]},
            feed=small_feed,
            progress_callback=callback,
        )
        assert len(results) == 2
        assert len(progress) == 2
        assert progress[0] == (1, 2)
        assert progress[1] == (2, 2)

    def test_different_objectives(self, small_feed: DataFeed) -> None:
        for obj in ("total_net_pnl", "win_rate", "profit_factor"):
            results = grid_search(
                strategy_factory=make_strategy,
                param_grid={"warmup": [10, 50]},
                feed=small_feed,
                objective=obj,
            )
            assert len(results) == 2

    def test_parallel_execution(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20, 30], "hold_period": [5, 10]},
            feed=small_feed,
            n_jobs=2,
        )
        assert len(results) == 6

    def test_parallel_matches_sequential(self, small_feed: DataFeed) -> None:
        param_grid = {"warmup": [10, 30], "hold_period": [5, 10]}
        seq = grid_search(
            strategy_factory=make_strategy,
            param_grid=param_grid,
            feed=small_feed,
            n_jobs=1,
        )
        par = grid_search(
            strategy_factory=make_strategy,
            param_grid=param_grid,
            feed=small_feed,
            n_jobs=2,
        )
        # Same number of results
        assert len(seq) == len(par)
        # Same best params (after sorting by objective)
        assert seq[0].params == par[0].params

    def test_custom_initial_capital(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10]},
            feed=small_feed,
            initial_capital=Decimal("500000"),
        )
        assert len(results) == 1

    def test_all_results_have_trades_list(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20]},
            feed=small_feed,
        )
        for r in results:
            assert isinstance(r.trades, list)


class TestGridSearchSummary:
    """Tests for grid_search_summary()."""

    def test_summary_format(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 50], "hold_period": [5]},
            feed=small_feed,
        )
        summary = grid_search_summary(results)
        assert len(summary) == 2
        first = summary[0]
        assert first["rank"] == 1
        assert "warmup" in first
        assert "hold_period" in first
        assert "net_pnl" in first
        assert "sharpe" in first
        assert "total_trades" in first
        assert "win_rate" in first
        assert "max_dd_pct" in first
        assert "profit_factor" in first
        assert "objective" in first

    def test_summary_rank_order(self, small_feed: DataFeed) -> None:
        results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 50, 100], "hold_period": [5]},
            feed=small_feed,
        )
        summary = grid_search_summary(results)
        ranks = [row["rank"] for row in summary]
        assert ranks == [1, 2, 3]

    def test_empty_results(self) -> None:
        summary = grid_search_summary([])
        assert summary == []


# ===================================================================
# optimizer.py (optuna)
# ===================================================================

# Skip all optuna tests if optuna is not installed.
has_optuna = True
try:
    import optuna
except ImportError:
    has_optuna = False

requires_optuna = pytest.mark.skipif(not has_optuna, reason="optuna not installed")


@requires_optuna
class TestParamSpec:
    """Tests for ParamSpec dataclass."""

    def test_int_param_defaults(self) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec

        spec = ParamSpec(name="warmup", low=5, high=50, param_type="int")
        assert spec.name == "warmup"
        assert spec.low == 5
        assert spec.high == 50
        assert spec.param_type == "int"
        assert spec.step is None
        assert spec.log is False

    def test_float_param_with_step(self) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec

        spec = ParamSpec(name="threshold", low=0.01, high=1.0, param_type="float", step=0.01)
        assert spec.param_type == "float"
        assert spec.step == 0.01

    def test_frozen(self) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec

        spec = ParamSpec(name="x", low=1, high=10)
        with pytest.raises(AttributeError):
            spec.name = "y"  # type: ignore[misc]

    def test_log_scale(self) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec

        spec = ParamSpec(name="lr", low=1e-5, high=1e-1, param_type="float", log=True)
        assert spec.log is True


@requires_optuna
class TestOptimize:
    """Tests for optimize()."""

    def test_basic_optimization(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=50, param_type="int"),
                ParamSpec(name="hold_period", low=3, high=20, param_type="int"),
            ],
            feed=small_feed,
            n_trials=10,
            objective="sharpe_ratio",
            seed=42,
        )
        assert result.n_trials == 10
        assert "warmup" in result.best_params
        assert "hold_period" in result.best_params
        assert isinstance(result.best_value, float)
        assert result.objective_name == "sharpe_ratio"
        assert result.direction == "maximize"

    def test_minimize_direction(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=30, param_type="int"),
            ],
            feed=small_feed,
            n_trials=5,
            objective="max_drawdown_pct",
            direction="minimize",
            seed=42,
        )
        assert result.direction == "minimize"

    def test_all_trials_recorded(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=20, param_type="int"),
            ],
            feed=small_feed,
            n_trials=8,
            seed=42,
        )
        assert len(result.all_trials) == 8
        for trial in result.all_trials:
            assert "number" in trial
            assert "params" in trial
            assert "value" in trial

    def test_progress_callback(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        progress: list[int] = []

        def callback(num: int, total: int, params: dict, value: float) -> None:
            progress.append(num)

        optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=20, param_type="int"),
            ],
            feed=small_feed,
            n_trials=5,
            seed=42,
            progress_callback=callback,
        )
        assert progress == [1, 2, 3, 4, 5]

    def test_result_frozen(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[ParamSpec(name="warmup", low=5, high=20, param_type="int")],
            feed=small_feed,
            n_trials=3,
            seed=42,
        )
        with pytest.raises(AttributeError):
            result.n_trials = 999  # type: ignore[misc]

    def test_with_step(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=10, high=50, step=10, param_type="int"),
            ],
            feed=small_feed,
            n_trials=5,
            seed=42,
        )
        # warmup should be multiples of 10 (step=10)
        for trial in result.all_trials:
            assert trial["params"]["warmup"] % 10 == 0

    def test_best_params_are_valid(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=30, param_type="int"),
                ParamSpec(name="hold_period", low=2, high=15, param_type="int"),
            ],
            feed=small_feed,
            n_trials=10,
            seed=42,
        )
        assert 5 <= result.best_params["warmup"] <= 30
        assert 2 <= result.best_params["hold_period"] <= 15

    def test_best_metrics_populated(self, small_feed: DataFeed) -> None:
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        result = optimize(
            strategy_factory=make_strategy,
            param_specs=[
                ParamSpec(name="warmup", low=5, high=20, param_type="int"),
            ],
            feed=small_feed,
            n_trials=5,
            seed=42,
        )
        assert isinstance(result.best_metrics, PerformanceMetrics)


# ===================================================================
# walk_forward.py
# ===================================================================


class TestCalculateWindowBoundaries:
    """Tests for _calculate_window_boundaries()."""

    def test_basic_boundaries(self) -> None:
        from datetime import datetime

        from taiex_backtest.optimization.walk_forward import _calculate_window_boundaries

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        boundaries = _calculate_window_boundaries(start, end, n_windows=3, in_sample_ratio=0.7)
        assert len(boundaries) == 3
        for is_start, is_end, oos_start, oos_end in boundaries:
            assert is_start < is_end
            assert oos_start < oos_end
            assert is_end <= oos_start or is_end == oos_start

    def test_single_window(self) -> None:
        from datetime import datetime

        from taiex_backtest.optimization.walk_forward import _calculate_window_boundaries

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        boundaries = _calculate_window_boundaries(start, end, n_windows=1, in_sample_ratio=0.7)
        assert len(boundaries) == 1

    def test_windows_progress_forward(self) -> None:
        from datetime import datetime

        from taiex_backtest.optimization.walk_forward import _calculate_window_boundaries

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        boundaries = _calculate_window_boundaries(start, end, n_windows=4, in_sample_ratio=0.6)
        # OOS windows should progress forward
        for i in range(len(boundaries) - 1):
            assert boundaries[i][2] < boundaries[i + 1][2]  # oos_start increases

    def test_full_coverage(self) -> None:
        from datetime import datetime

        from taiex_backtest.optimization.walk_forward import _calculate_window_boundaries

        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        boundaries = _calculate_window_boundaries(start, end, n_windows=2, in_sample_ratio=0.7)
        # Last OOS end should be close to or equal to the overall end
        last_oos_end = boundaries[-1][3]
        assert last_oos_end <= end


class TestWalkForward:
    """Tests for walk_forward()."""

    def test_basic_walk_forward(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import WalkForwardResult, walk_forward

        result = walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 30], "hold_period": [5, 10]},
            feed=medium_feed,
            n_windows=2,
            in_sample_ratio=0.7,
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_windows >= 1
        assert result.in_sample_ratio == 0.7
        assert result.objective == "sharpe_ratio"

    def test_window_results(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import WalkForwardWindow, walk_forward

        result = walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20], "hold_period": [5]},
            feed=medium_feed,
            n_windows=2,
        )
        for w in result.windows:
            assert isinstance(w, WalkForwardWindow)
            assert "warmup" in w.best_params
            assert w.in_sample_start < w.in_sample_end

    def test_invalid_n_windows(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import walk_forward

        with pytest.raises(ValueError, match="n_windows"):
            walk_forward(
                strategy_factory=make_strategy,
                param_grid={"warmup": [10]},
                feed=medium_feed,
                n_windows=0,
            )

    def test_invalid_ratio(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import walk_forward

        with pytest.raises(ValueError, match="in_sample_ratio"):
            walk_forward(
                strategy_factory=make_strategy,
                param_grid={"warmup": [10]},
                feed=medium_feed,
                in_sample_ratio=0.05,
            )

    def test_progress_callback(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import walk_forward

        phases: list[tuple[int, str]] = []

        def callback(window_num: int, total: int, phase: str) -> None:
            phases.append((window_num, phase))

        walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20]},
            feed=medium_feed,
            n_windows=2,
            progress_callback=callback,
        )
        assert any(p[1] == "in_sample" for p in phases)
        assert any(p[1] == "out_of_sample" for p in phases)

    def test_oos_metrics_populated(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import walk_forward

        result = walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20], "hold_period": [5]},
            feed=medium_feed,
            n_windows=2,
        )
        for w in result.windows:
            assert isinstance(w.out_of_sample_metrics, PerformanceMetrics)


class TestWalkForwardSummary:
    """Tests for walk_forward_summary()."""

    def test_summary_format(self, medium_feed: DataFeed) -> None:
        from taiex_backtest.optimization.walk_forward import walk_forward, walk_forward_summary

        result = walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 30], "hold_period": [5]},
            feed=medium_feed,
            n_windows=2,
        )
        summary = walk_forward_summary(result)
        assert len(summary) == result.n_windows
        if summary:
            row = summary[0]
            assert "window" in row
            assert "is_period" in row
            assert "oos_period" in row
            assert "best_params" in row
            assert "oos_sharpe" in row

    def test_empty_summary(self) -> None:
        from taiex_backtest.optimization.walk_forward import WalkForwardResult, walk_forward_summary

        result = WalkForwardResult(
            windows=[],
            combined_trades=[],
            combined_metrics=_empty_metrics(),
            n_windows=0,
            in_sample_ratio=0.7,
            objective="sharpe_ratio",
        )
        summary = walk_forward_summary(result)
        assert summary == []


# ===================================================================
# Integration tests
# ===================================================================


class TestOptimizationIntegration:
    """Integration tests combining multiple optimization modules."""

    def test_grid_search_then_walk_forward(self, medium_feed: DataFeed) -> None:
        """Grid search followed by walk-forward validation."""
        from taiex_backtest.optimization.walk_forward import walk_forward

        # First find promising parameter ranges with grid search
        gs_results = grid_search(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20, 30], "hold_period": [5, 10]},
            feed=medium_feed,
        )
        assert len(gs_results) == 6

        # Then validate with walk-forward
        wf_result = walk_forward(
            strategy_factory=make_strategy,
            param_grid={"warmup": [10, 20, 30], "hold_period": [5, 10]},
            feed=medium_feed,
            n_windows=2,
        )
        assert wf_result.n_windows >= 1

    def test_grid_search_best_is_reproducible(self, small_feed: DataFeed) -> None:
        """Same parameters should produce same results."""
        param_grid = {"warmup": [10, 20], "hold_period": [5]}
        r1 = grid_search(
            strategy_factory=make_strategy,
            param_grid=param_grid,
            feed=small_feed,
        )
        r2 = grid_search(
            strategy_factory=make_strategy,
            param_grid=param_grid,
            feed=small_feed,
        )
        assert r1[0].params == r2[0].params
        assert r1[0].objective_value == r2[0].objective_value

    @pytest.mark.skipif(not has_optuna, reason="optuna not installed")
    def test_optimizer_reproducible_with_seed(self, small_feed: DataFeed) -> None:
        """Same seed should produce same optimization results."""
        from taiex_backtest.optimization.optimizer import ParamSpec, optimize

        specs = [ParamSpec(name="warmup", low=5, high=30, param_type="int")]
        r1 = optimize(
            strategy_factory=make_strategy,
            param_specs=specs,
            feed=small_feed,
            n_trials=5,
            seed=42,
        )
        r2 = optimize(
            strategy_factory=make_strategy,
            param_specs=specs,
            feed=small_feed,
            n_trials=5,
            seed=42,
        )
        assert r1.best_params == r2.best_params
        assert r1.best_value == r2.best_value
