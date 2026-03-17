"""Main backtesting engine orchestrating all components."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from ..domain.enums import EventType, OrderType, ProductType, Side
from ..domain.events import Event, TickEvent
from ..domain.models import Fill, Order, Tick
from ..data.feed import DataFeed
from ..strategy.base import Strategy
from .clock import Clock
from .commission import CommissionCalculator
from .event_bus import EventBus
from .matching_engine import MatchingEngine
from .position_tracker import PositionTracker


class BacktestEngine:
    """Main backtesting engine that orchestrates the simulation."""

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        product: ProductType = ProductType.TX,
        initial_capital: Decimal = Decimal("1000000"),
    ):
        self._feed = feed
        self._strategy = strategy
        self._product = product
        self._initial_capital = initial_capital
        self._capital = initial_capital

        # Core components
        self._clock = Clock()
        self._event_bus = EventBus()
        self._commission_calc = CommissionCalculator()
        self._matching_engine = MatchingEngine(self._commission_calc)
        self._position_tracker = PositionTracker()

        # State
        self._running = False
        self._tick_count = 0
        self._fills: list[Fill] = []

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def position_tracker(self) -> PositionTracker:
        return self._position_tracker

    @property
    def capital(self) -> Decimal:
        return self._capital

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def trades(self):
        return self._position_tracker.trades

    def run(self) -> None:
        """Run the backtest simulation."""
        self._running = True
        self._publish_event(EventType.ENGINE_START)

        # Initialize strategy
        ctx = self._create_context()
        self._strategy.on_init(ctx)

        last_tick: Tick | None = None
        for tick in self._feed.iter_ticks():
            last_tick = tick
            if not self._running:
                break

            self._tick_count += 1
            self._clock.update(tick.timestamp)

            # Process pending orders against this tick
            fills = self._matching_engine.process_tick(
                tick.price, tick.timestamp, tick.product
            )
            for fill in fills:
                self._process_fill(fill)

            # Update unrealized PnL
            self._position_tracker.update_unrealized_pnl(
                self._product, tick.price
            )

            # Publish tick event
            tick_event = TickEvent(
                event_type=EventType.TICK,
                timestamp=tick.timestamp,
                price=tick.price,
                volume=tick.volume,
                product=tick.product,
            )
            self._event_bus.publish(tick_event)

            # Call strategy
            ctx = self._create_context(tick)
            self._strategy.on_tick(ctx, tick)

        # Finalize: let strategy submit closing orders
        self._strategy.on_stop(self._create_context(last_tick))

        # Process any orders submitted during on_stop (e.g. close_position)
        if self._clock.now is not None and last_tick is not None:
            last_price = last_tick.price
            last_ts = self._clock.now
            fills = self._matching_engine.process_tick(last_price, last_ts, self._product)
            for fill in fills:
                self._process_fill(fill)

        self._publish_event(EventType.ENGINE_STOP)
        self._running = False

    def stop(self) -> None:
        """Stop the backtesting engine."""
        self._running = False

    def submit_order(
        self,
        side: Side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tag: str = "",
    ) -> Order:
        """Submit an order to the matching engine."""
        order = Order(
            id=uuid4(),
            timestamp=self._clock.now or datetime.now(),
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=self._product,
            price=price,
            stop_price=stop_price,
            tag=tag,
        )
        return self._matching_engine.submit_order(order)

    def cancel_all_orders(self) -> list[Order]:
        """Cancel all pending orders."""
        return self._matching_engine.cancel_all()

    def _process_fill(self, fill: Fill) -> None:
        """Process a fill: update position and capital."""
        self._fills.append(fill)

        trade = self._position_tracker.process_fill(fill)
        if trade is not None:
            self._capital += trade.net_pnl

        self._event_bus.publish(Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=fill.timestamp,
            data={
                "order_id": str(fill.order_id),
                "side": fill.side.name,
                "price": str(fill.price),
                "quantity": fill.quantity,
            },
        ))

    def _create_context(self, tick: Tick | None = None):
        """Create a strategy context."""
        from ..strategy.context import StrategyContext
        return StrategyContext(engine=self, tick=tick)

    def _publish_event(self, event_type: EventType) -> None:
        """Publish a simple event."""
        self._event_bus.publish(Event(
            event_type=event_type,
            timestamp=self._clock.now or datetime.now(),
        ))
