"""Enumerations for the backtesting system."""

from enum import Enum, auto


class Side(Enum):
    """Order/trade side."""
    BUY = auto()
    SELL = auto()


class OrderType(Enum):
    """Order type."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()


class ProductType(Enum):
    """TAIEX futures product types."""
    TX = "TX"      # 大台 (Full-size TAIEX futures)
    MTX = "MTX"    # 小台 (Mini-TAIEX futures)
    XMT = "XMT"    # 微台 (Micro-TAIEX futures)


class Session(Enum):
    """Trading session."""
    DAY = auto()    # 日盤 08:45-13:45
    NIGHT = auto()  # 夜盤 15:00-05:00


class EventType(Enum):
    """Event bus event types."""
    TICK = auto()
    ORDER_SUBMITTED = auto()
    ORDER_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    POSITION_CHANGED = auto()
    TRADE_CLOSED = auto()
    SESSION_START = auto()
    SESSION_END = auto()
    ENGINE_START = auto()
    ENGINE_STOP = auto()
