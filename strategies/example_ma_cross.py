"""Example: Moving Average Crossover Strategy for TAIEX futures."""

from decimal import Decimal

from taiex_backtest.domain.models import Tick
from taiex_backtest.strategy.base import Strategy
from taiex_backtest.strategy.context import StrategyContext
from taiex_backtest.strategy.signal import CrossDetector, MovingAverage


class MACrossStrategy(Strategy):
    """Simple moving average crossover strategy.

    Buys when fast MA crosses above slow MA.
    Sells when fast MA crosses below slow MA.
    """

    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        quantity: int = 1,
    ):
        self._fast_ma = MovingAverage(fast_period)
        self._slow_ma = MovingAverage(slow_period)
        self._cross = CrossDetector()
        self._quantity = quantity

    @property
    def name(self) -> str:
        return f"MACross({self._fast_ma.period},{self._slow_ma.period})"

    def on_init(self, ctx: StrategyContext) -> None:
        """Initialize strategy state."""

    def on_tick(self, ctx: StrategyContext, tick: Tick) -> None:
        """Process each tick."""
        fast_val = self._fast_ma.update(tick.price)
        slow_val = self._slow_ma.update(tick.price)

        if fast_val is None or slow_val is None:
            return

        crossover, crossunder = self._cross.update(fast_val, slow_val)

        if crossover and ctx.position.is_flat:
            ctx.buy(quantity=self._quantity, tag="ma_cross_buy")

        elif crossunder and not ctx.position.is_flat:
            ctx.close_position(tag="ma_cross_sell")

    def on_stop(self, ctx: StrategyContext) -> None:
        """Close any remaining position."""
        if not ctx.position.is_flat:
            ctx.close_position(tag="ma_cross_exit")
