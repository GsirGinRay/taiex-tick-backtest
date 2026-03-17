"""Strategy context providing controlled access to engine capabilities."""

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from ..domain.enums import OrderType, ProductType, Side
from ..domain.models import Order, Position, Tick, Trade

if TYPE_CHECKING:
    from ..engine.backtest_engine import BacktestEngine


class StrategyContext:
    """Provides a safe interface for strategies to interact with the engine."""

    def __init__(
        self,
        engine: "BacktestEngine",
        tick: Optional[Tick] = None,
    ):
        self._engine = engine
        self._tick = tick

    @property
    def tick(self) -> Optional[Tick]:
        """Current tick data."""
        return self._tick

    @property
    def position(self) -> Position:
        """Current position."""
        return self._engine.position_tracker.get_position(self._engine._product)

    @property
    def capital(self) -> Decimal:
        """Current available capital."""
        return self._engine.capital

    @property
    def trades(self) -> list[Trade]:
        """List of completed trades."""
        return self._engine.trades

    @property
    def tick_count(self) -> int:
        """Number of ticks processed."""
        return self._engine.clock.tick_count

    def buy(
        self,
        quantity: int = 1,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tag: str = "",
    ) -> Order:
        """Submit a buy order."""
        return self._engine.submit_order(
            side=Side.BUY,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            tag=tag,
        )

    def sell(
        self,
        quantity: int = 1,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tag: str = "",
    ) -> Order:
        """Submit a sell order."""
        return self._engine.submit_order(
            side=Side.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            tag=tag,
        )

    def close_position(self, tag: str = "") -> Optional[Order]:
        """Close the current position with a market order."""
        pos = self.position
        if pos.is_flat:
            return None

        if pos.side == Side.BUY:
            return self.sell(quantity=pos.quantity, tag=tag)
        return self.buy(quantity=pos.quantity, tag=tag)

    def cancel_all(self) -> list[Order]:
        """Cancel all pending orders."""
        return self._engine.cancel_all_orders()
