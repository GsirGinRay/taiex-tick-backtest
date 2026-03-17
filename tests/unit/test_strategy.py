"""Tests for strategy framework: signals, context, base."""

from decimal import Decimal

import pytest

from taiex_backtest.strategy.signal import (
    CrossDetector,
    ExponentialMovingAverage,
    MovingAverage,
)
from taiex_backtest.strategy.registry import StrategyRegistry


class TestMovingAverage:
    def test_not_ready(self):
        ma = MovingAverage(3)
        ma.update(Decimal("10"))
        assert not ma.is_ready
        assert ma.value is None

    def test_ready_after_period(self):
        ma = MovingAverage(3)
        ma.update(Decimal("10"))
        ma.update(Decimal("20"))
        val = ma.update(Decimal("30"))
        assert ma.is_ready
        assert val == Decimal("20")

    def test_sliding_window(self):
        ma = MovingAverage(3)
        ma.update(Decimal("10"))
        ma.update(Decimal("20"))
        ma.update(Decimal("30"))
        val = ma.update(Decimal("40"))
        # (20 + 30 + 40) / 3 = 30
        assert val == Decimal("30")

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            MovingAverage(0)

    def test_reset(self):
        ma = MovingAverage(3)
        ma.update(Decimal("10"))
        ma.update(Decimal("20"))
        ma.update(Decimal("30"))
        ma.reset()
        assert not ma.is_ready


class TestExponentialMovingAverage:
    def test_not_ready(self):
        ema = ExponentialMovingAverage(3)
        ema.update(Decimal("10"))
        assert not ema.is_ready

    def test_initial_value_is_sma(self):
        ema = ExponentialMovingAverage(3)
        ema.update(Decimal("10"))
        ema.update(Decimal("20"))
        val = ema.update(Decimal("30"))
        # Initial EMA = SMA = (10+20+30)/3 = 20
        assert val == Decimal("20")

    def test_subsequent_values(self):
        ema = ExponentialMovingAverage(3)
        ema.update(Decimal("10"))
        ema.update(Decimal("20"))
        ema.update(Decimal("30"))
        val = ema.update(Decimal("40"))
        assert val is not None
        # EMA should be > 20 (previous) and < 40 (current)
        assert Decimal("20") < val < Decimal("40")

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            ExponentialMovingAverage(0)


class TestCrossDetector:
    def test_crossover(self):
        cd = CrossDetector()
        cd.update(Decimal("10"), Decimal("20"))
        crossover, crossunder = cd.update(Decimal("25"), Decimal("20"))
        assert crossover
        assert not crossunder

    def test_crossunder(self):
        cd = CrossDetector()
        cd.update(Decimal("25"), Decimal("20"))
        crossover, crossunder = cd.update(Decimal("15"), Decimal("20"))
        assert not crossover
        assert crossunder

    def test_no_cross(self):
        cd = CrossDetector()
        cd.update(Decimal("10"), Decimal("20"))
        crossover, crossunder = cd.update(Decimal("15"), Decimal("20"))
        assert not crossover
        assert not crossunder

    def test_first_update_no_cross(self):
        cd = CrossDetector()
        crossover, crossunder = cd.update(Decimal("10"), Decimal("20"))
        assert not crossover
        assert not crossunder


class TestStrategyRegistry:
    def test_register_and_get(self):
        reg = StrategyRegistry()
        from taiex_backtest.strategy.base import Strategy
        from taiex_backtest.domain.models import Tick

        @reg.register("test_strat")
        class TestStrat(Strategy):
            def on_tick(self, ctx, tick: Tick):
                pass

        cls = reg.get("test_strat")
        assert cls is TestStrat

    def test_list_strategies(self):
        reg = StrategyRegistry()
        from taiex_backtest.strategy.base import Strategy
        from taiex_backtest.domain.models import Tick

        @reg.register("a")
        class A(Strategy):
            def on_tick(self, ctx, tick: Tick):
                pass

        @reg.register("b")
        class B(Strategy):
            def on_tick(self, ctx, tick: Tick):
                pass

        assert sorted(reg.list_strategies()) == ["a", "b"]

    def test_get_unknown(self):
        reg = StrategyRegistry()
        with pytest.raises(KeyError):
            reg.get("unknown")
