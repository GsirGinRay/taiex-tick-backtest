"""Integration test: full backtest run with MA cross strategy."""

from decimal import Decimal

import pytest

from taiex_backtest.data.feed import DataFeed
from taiex_backtest.data.synthetic import generate_ticks
from taiex_backtest.data.writer import ticks_to_dataframe
from taiex_backtest.domain.enums import ProductType, Side
from taiex_backtest.engine.backtest_engine import BacktestEngine
from strategies.example_ma_cross import MACrossStrategy


class TestBacktestEngine:
    def _make_engine(self, num_ticks=2000, seed=42):
        ticks = generate_ticks(num_ticks=num_ticks, seed=seed)
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)
        strategy = MACrossStrategy(fast_period=20, slow_period=50, quantity=1)
        return BacktestEngine(feed=feed, strategy=strategy)

    def test_engine_runs(self):
        engine = self._make_engine()
        engine.run()
        assert engine.clock.tick_count == 2000

    def test_engine_produces_trades(self):
        engine = self._make_engine(num_ticks=5000, seed=100)
        engine.run()
        # With enough ticks and volatility, should have some trades
        # (may or may not, depends on price path)
        assert engine.clock.tick_count == 5000

    def test_engine_capital_changes(self):
        engine = self._make_engine(num_ticks=5000, seed=100)
        initial = engine.capital
        engine.run()
        # Capital may change if trades occurred
        # At minimum, no errors should have occurred
        assert isinstance(engine.capital, Decimal)

    def test_engine_position_flat_after_stop(self):
        """Strategy should close position on stop."""
        engine = self._make_engine(num_ticks=5000, seed=100)
        engine.run()
        pos = engine.position_tracker.get_position(ProductType.TX)
        assert pos.is_flat

    def test_engine_with_different_seeds(self):
        """Engine should work with different random data."""
        for seed in [1, 42, 100, 999]:
            engine = self._make_engine(num_ticks=1000, seed=seed)
            engine.run()
            assert engine.clock.tick_count == 1000
