"""Domain events for the event-driven architecture."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from .enums import EventType, ProductType, Side


@dataclass(frozen=True)
class Event:
    """Base event."""
    event_type: EventType
    timestamp: datetime
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class TickEvent(Event):
    """Tick data event."""
    price: Decimal = Decimal("0")
    volume: int = 0
    product: ProductType = ProductType.TX

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.TICK)


@dataclass(frozen=True)
class OrderEvent(Event):
    """Order lifecycle event."""
    order_id: UUID | None = None
    side: Side | None = None
    price: Decimal = Decimal("0")
    quantity: int = 0

    def __post_init__(self):
        if self.event_type not in (
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
        ):
            object.__setattr__(self, 'event_type', EventType.ORDER_SUBMITTED)


@dataclass(frozen=True)
class PositionEvent(Event):
    """Position change event."""
    product: ProductType = ProductType.TX
    quantity: int = 0
    avg_price: Decimal = Decimal("0")

    def __post_init__(self):
        object.__setattr__(self, 'event_type', EventType.POSITION_CHANGED)
