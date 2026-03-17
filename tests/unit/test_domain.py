"""Tests for domain models, enums, and events."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from taiex_backtest.domain.enums import (
    EventType, OrderStatus, OrderType, ProductType, Session, Side,
)
from taiex_backtest.domain.models import (
    CONTRACT_SPECS, ContractSpec, Fill, Order, Position, Tick, Trade,
    TX_SPEC, MTX_SPEC, XMT_SPEC,
)
from taiex_backtest.domain.events import Event, TickEvent, OrderEvent, PositionEvent
from taiex_backtest.domain.errors import (
    BacktestError, InsufficientMarginError, InvalidOrderError, DataError,
)


class TestEnums:
    def test_side_values(self):
        assert Side.BUY is not Side.SELL

    def test_order_types(self):
        assert OrderType.MARKET is not None
        assert OrderType.LIMIT is not None
        assert OrderType.STOP is not None

    def test_product_types(self):
        assert ProductType.TX.value == "TX"
        assert ProductType.MTX.value == "MTX"
        assert ProductType.XMT.value == "XMT"

    def test_session(self):
        assert Session.DAY is not Session.NIGHT


class TestContractSpec:
    def test_tx_spec(self):
        assert TX_SPEC.product == ProductType.TX
        assert TX_SPEC.point_value == Decimal("200")
        assert TX_SPEC.tick_size == Decimal("1")

    def test_mtx_spec(self):
        assert MTX_SPEC.product == ProductType.MTX
        assert MTX_SPEC.point_value == Decimal("50")

    def test_xmt_spec(self):
        assert XMT_SPEC.product == ProductType.XMT
        assert XMT_SPEC.point_value == Decimal("10")

    def test_tick_value(self):
        assert TX_SPEC.tick_value == Decimal("200")
        assert MTX_SPEC.tick_value == Decimal("50")
        assert XMT_SPEC.tick_value == Decimal("10")

    def test_contract_specs_dict(self):
        assert CONTRACT_SPECS[ProductType.TX] == TX_SPEC
        assert CONTRACT_SPECS[ProductType.MTX] == MTX_SPEC
        assert CONTRACT_SPECS[ProductType.XMT] == XMT_SPEC

    def test_frozen(self):
        with pytest.raises(AttributeError):
            TX_SPEC.point_value = Decimal("100")


class TestTick:
    def test_create_tick(self):
        ts = datetime(2024, 1, 2, 9, 0, 0)
        tick = Tick(
            timestamp=ts,
            price=Decimal("20000"),
            volume=5,
        )
        assert tick.timestamp == ts
        assert tick.price == Decimal("20000")
        assert tick.volume == 5
        assert tick.product == ProductType.TX
        assert tick.session == Session.DAY

    def test_tick_frozen(self):
        tick = Tick(
            timestamp=datetime.now(),
            price=Decimal("20000"),
            volume=1,
        )
        with pytest.raises(AttributeError):
            tick.price = Decimal("19999")


class TestOrder:
    def test_create_market_order(self):
        oid = uuid4()
        order = Order(
            id=oid,
            timestamp=datetime.now(),
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )
        assert order.id == oid
        assert order.side == Side.BUY
        assert order.quantity == 1
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0

    def test_create_limit_order(self):
        order = Order(
            id=uuid4(),
            timestamp=datetime.now(),
            side=Side.SELL,
            quantity=2,
            order_type=OrderType.LIMIT,
            price=Decimal("20100"),
        )
        assert order.price == Decimal("20100")


class TestPosition:
    def test_flat_position(self):
        pos = Position(
            product=ProductType.TX,
            side=None,
            quantity=0,
            avg_price=Decimal("0"),
        )
        assert pos.is_flat

    def test_long_position(self):
        pos = Position(
            product=ProductType.TX,
            side=Side.BUY,
            quantity=2,
            avg_price=Decimal("20000"),
        )
        assert not pos.is_flat
        assert pos.side == Side.BUY


class TestTrade:
    def test_trade_pnl(self):
        trade = Trade(
            id=uuid4(),
            product=ProductType.TX,
            side=Side.BUY,
            entry_time=datetime(2024, 1, 2, 9, 0),
            entry_price=Decimal("20000"),
            exit_time=datetime(2024, 1, 2, 10, 0),
            exit_price=Decimal("20050"),
            quantity=1,
            pnl=Decimal("10000"),
            commission=Decimal("120"),
            tax=Decimal("16"),
        )
        assert trade.net_pnl == Decimal("10000") - Decimal("120") - Decimal("16")
        assert trade.points == Decimal("50")

    def test_short_trade_points(self):
        trade = Trade(
            id=uuid4(),
            product=ProductType.TX,
            side=Side.SELL,
            entry_time=datetime(2024, 1, 2, 9, 0),
            entry_price=Decimal("20050"),
            exit_time=datetime(2024, 1, 2, 10, 0),
            exit_price=Decimal("20000"),
            quantity=1,
            pnl=Decimal("10000"),
            commission=Decimal("120"),
            tax=Decimal("16"),
        )
        assert trade.points == Decimal("50")

    def test_holding_time(self):
        from datetime import timedelta
        trade = Trade(
            id=uuid4(),
            product=ProductType.TX,
            side=Side.BUY,
            entry_time=datetime(2024, 1, 2, 9, 0),
            entry_price=Decimal("20000"),
            exit_time=datetime(2024, 1, 2, 10, 30),
            exit_price=Decimal("20050"),
            quantity=1,
            pnl=Decimal("10000"),
            commission=Decimal("0"),
            tax=Decimal("0"),
        )
        assert trade.holding_time == timedelta(hours=1, minutes=30)


class TestEvents:
    def test_base_event(self):
        event = Event(
            event_type=EventType.ENGINE_START,
            timestamp=datetime.now(),
        )
        assert event.event_type == EventType.ENGINE_START

    def test_tick_event(self):
        event = TickEvent(
            event_type=EventType.TICK,
            timestamp=datetime.now(),
            price=Decimal("20000"),
            volume=5,
        )
        assert event.event_type == EventType.TICK
        assert event.price == Decimal("20000")


class TestErrors:
    def test_backtest_error_hierarchy(self):
        assert issubclass(InsufficientMarginError, BacktestError)
        assert issubclass(InvalidOrderError, BacktestError)
        assert issubclass(DataError, BacktestError)

    def test_raise_invalid_order(self):
        with pytest.raises(InvalidOrderError):
            raise InvalidOrderError("bad order")
