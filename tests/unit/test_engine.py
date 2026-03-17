"""Tests for engine components: event bus, clock, commission, matching, position."""

from datetime import datetime, time
from decimal import Decimal
from uuid import uuid4

import pytest

from taiex_backtest.domain.enums import (
    OrderStatus, OrderType, ProductType, Session, Side,
)
from taiex_backtest.domain.errors import InvalidOrderError
from taiex_backtest.domain.events import Event, TickEvent
from taiex_backtest.domain.models import Fill, Order, TX_SPEC
from taiex_backtest.engine.event_bus import EventBus
from taiex_backtest.engine.clock import Clock
from taiex_backtest.engine.commission import CommissionCalculator
from taiex_backtest.engine.matching_engine import MatchingEngine
from taiex_backtest.engine.position_tracker import PositionTracker


class TestEventBus:
    def test_subscribe_and_publish(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        from taiex_backtest.domain.enums import EventType
        bus.subscribe(EventType.TICK, handler)
        event = Event(event_type=EventType.TICK, timestamp=datetime.now())
        bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    def test_unsubscribe(self):
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event)

        from taiex_backtest.domain.enums import EventType
        bus.subscribe(EventType.TICK, handler)
        bus.unsubscribe(EventType.TICK, handler)
        bus.publish(Event(event_type=EventType.TICK, timestamp=datetime.now()))
        assert len(received) == 0

    def test_multiple_handlers(self):
        bus = EventBus()
        r1, r2 = [], []

        from taiex_backtest.domain.enums import EventType
        bus.subscribe(EventType.TICK, lambda e: r1.append(e))
        bus.subscribe(EventType.TICK, lambda e: r2.append(e))
        bus.publish(Event(event_type=EventType.TICK, timestamp=datetime.now()))
        assert len(r1) == 1
        assert len(r2) == 1

    def test_clear(self):
        bus = EventBus()
        received = []
        from taiex_backtest.domain.enums import EventType
        bus.subscribe(EventType.TICK, lambda e: received.append(e))
        bus.clear()
        bus.publish(Event(event_type=EventType.TICK, timestamp=datetime.now()))
        assert len(received) == 0


class TestClock:
    def test_initial_state(self):
        clock = Clock()
        assert clock.now is None
        assert clock.tick_count == 0

    def test_update(self):
        clock = Clock()
        ts = datetime(2024, 1, 2, 9, 0, 0)
        clock.update(ts)
        assert clock.now == ts
        assert clock.tick_count == 1

    def test_cannot_go_backwards(self):
        clock = Clock()
        clock.update(datetime(2024, 1, 2, 10, 0, 0))
        with pytest.raises(ValueError):
            clock.update(datetime(2024, 1, 2, 9, 0, 0))

    def test_get_session_day(self):
        ts = datetime(2024, 1, 2, 9, 30, 0)
        assert Clock.get_session(ts) == Session.DAY

    def test_get_session_night(self):
        ts = datetime(2024, 1, 2, 16, 0, 0)
        assert Clock.get_session(ts) == Session.NIGHT

    def test_is_trading_hours(self):
        assert Clock.is_trading_hours(datetime(2024, 1, 2, 9, 0, 0))
        assert Clock.is_trading_hours(datetime(2024, 1, 2, 16, 0, 0))
        assert not Clock.is_trading_hours(datetime(2024, 1, 2, 7, 0, 0))
        assert not Clock.is_trading_hours(datetime(2024, 1, 2, 14, 0, 0))

    def test_reset(self):
        clock = Clock()
        clock.update(datetime(2024, 1, 2, 9, 0))
        clock.reset()
        assert clock.now is None
        assert clock.tick_count == 0


class TestCommissionCalculator:
    def test_tx_commission(self):
        calc = CommissionCalculator()
        comm = calc.calculate_commission(ProductType.TX, 1)
        assert comm == Decimal("60")

    def test_tx_commission_multi(self):
        calc = CommissionCalculator()
        comm = calc.calculate_commission(ProductType.TX, 3)
        assert comm == Decimal("180")

    def test_tax(self):
        calc = CommissionCalculator()
        tax = calc.calculate_tax(ProductType.TX, Decimal("20000"), 1)
        # notional = 20000 * 200 * 1 = 4,000,000
        # tax = 4,000,000 * 0.00002 = 80
        assert tax == Decimal("80")

    def test_total_cost(self):
        calc = CommissionCalculator()
        total = calc.calculate_total_cost(ProductType.TX, Decimal("20000"), 1)
        assert total == Decimal("60") + Decimal("80")


class TestMatchingEngine:
    def _make_order(self, side=Side.BUY, order_type=OrderType.MARKET, **kwargs):
        return Order(
            id=uuid4(),
            timestamp=datetime.now(),
            side=side,
            quantity=kwargs.get("quantity", 1),
            order_type=order_type,
            product=kwargs.get("product", ProductType.TX),
            price=kwargs.get("price"),
            stop_price=kwargs.get("stop_price"),
        )

    def test_market_order_fill(self):
        engine = MatchingEngine()
        order = self._make_order(side=Side.BUY)
        engine.submit_order(order)
        fills = engine.process_tick(Decimal("20000"), datetime.now())
        assert len(fills) == 1
        assert fills[0].price == Decimal("20000")
        assert fills[0].side == Side.BUY

    def test_limit_buy_fill(self):
        engine = MatchingEngine()
        order = self._make_order(
            side=Side.BUY, order_type=OrderType.LIMIT, price=Decimal("20000")
        )
        engine.submit_order(order)

        # Price too high, no fill
        fills = engine.process_tick(Decimal("20050"), datetime.now())
        assert len(fills) == 0

        # Price at limit, should fill
        fills = engine.process_tick(Decimal("20000"), datetime.now())
        assert len(fills) == 1

    def test_limit_sell_fill(self):
        engine = MatchingEngine()
        order = self._make_order(
            side=Side.SELL, order_type=OrderType.LIMIT, price=Decimal("20100")
        )
        engine.submit_order(order)

        fills = engine.process_tick(Decimal("20050"), datetime.now())
        assert len(fills) == 0

        fills = engine.process_tick(Decimal("20100"), datetime.now())
        assert len(fills) == 1

    def test_stop_buy_fill(self):
        engine = MatchingEngine()
        order = self._make_order(
            side=Side.BUY, order_type=OrderType.STOP, stop_price=Decimal("20100")
        )
        engine.submit_order(order)

        fills = engine.process_tick(Decimal("20050"), datetime.now())
        assert len(fills) == 0

        fills = engine.process_tick(Decimal("20100"), datetime.now())
        assert len(fills) == 1

    def test_cancel_order(self):
        engine = MatchingEngine()
        order = self._make_order(order_type=OrderType.LIMIT, price=Decimal("19000"))
        submitted = engine.submit_order(order)
        cancelled = engine.cancel_order(submitted.id)
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED
        assert len(engine.pending_orders) == 0

    def test_cancel_all(self):
        engine = MatchingEngine()
        engine.submit_order(self._make_order(order_type=OrderType.LIMIT, price=Decimal("19000")))
        engine.submit_order(self._make_order(order_type=OrderType.LIMIT, price=Decimal("19500")))
        cancelled = engine.cancel_all()
        assert len(cancelled) == 2
        assert len(engine.pending_orders) == 0

    def test_invalid_order_quantity(self):
        engine = MatchingEngine()
        order = self._make_order(quantity=0)
        with pytest.raises(InvalidOrderError):
            engine.submit_order(order)

    def test_invalid_limit_no_price(self):
        engine = MatchingEngine()
        order = self._make_order(order_type=OrderType.LIMIT, price=None)
        with pytest.raises(InvalidOrderError):
            engine.submit_order(order)


class TestPositionTracker:
    def _make_fill(self, side=Side.BUY, price="20000", quantity=1, **kwargs):
        return Fill(
            order_id=uuid4(),
            timestamp=kwargs.get("timestamp", datetime.now()),
            side=side,
            price=Decimal(price),
            quantity=quantity,
            product=kwargs.get("product", ProductType.TX),
            commission=kwargs.get("commission", Decimal("60")),
            tax=kwargs.get("tax", Decimal("80")),
        )

    def test_open_position(self):
        tracker = PositionTracker()
        fill = self._make_fill(side=Side.BUY, price="20000")
        trade = tracker.process_fill(fill)
        assert trade is None  # No trade closed
        pos = tracker.get_position(ProductType.TX)
        assert pos.side == Side.BUY
        assert pos.quantity == 1
        assert pos.avg_price == Decimal("20000")

    def test_close_position(self):
        tracker = PositionTracker()
        tracker.process_fill(self._make_fill(side=Side.BUY, price="20000"))
        trade = tracker.process_fill(self._make_fill(side=Side.SELL, price="20050"))
        assert trade is not None
        assert trade.pnl == Decimal("50") * Decimal("200")  # 50 points * 200 TWD/point
        pos = tracker.get_position(ProductType.TX)
        assert pos.is_flat

    def test_add_to_position(self):
        tracker = PositionTracker()
        tracker.process_fill(self._make_fill(side=Side.BUY, price="20000"))
        tracker.process_fill(self._make_fill(side=Side.BUY, price="20100"))
        pos = tracker.get_position(ProductType.TX)
        assert pos.quantity == 2
        assert pos.avg_price == Decimal("20050")

    def test_short_position(self):
        tracker = PositionTracker()
        tracker.process_fill(self._make_fill(side=Side.SELL, price="20100"))
        trade = tracker.process_fill(self._make_fill(side=Side.BUY, price="20000"))
        assert trade is not None
        assert trade.pnl == Decimal("100") * Decimal("200")  # 100 points profit
        assert trade.side == Side.SELL

    def test_unrealized_pnl(self):
        tracker = PositionTracker()
        tracker.process_fill(self._make_fill(side=Side.BUY, price="20000"))
        pos = tracker.update_unrealized_pnl(ProductType.TX, Decimal("20050"))
        assert pos.unrealized_pnl == Decimal("50") * Decimal("200")

    def test_reset(self):
        tracker = PositionTracker()
        tracker.process_fill(self._make_fill(side=Side.BUY, price="20000"))
        tracker.reset()
        pos = tracker.get_position(ProductType.TX)
        assert pos.is_flat
        assert len(tracker.trades) == 0
