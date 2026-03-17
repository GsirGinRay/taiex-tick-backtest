"""Immutable domain models."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from .enums import OrderStatus, OrderType, ProductType, Session, Side


@dataclass(frozen=True)
class Tick:
    """A single tick data point."""
    timestamp: datetime
    price: Decimal
    volume: int
    product: ProductType = ProductType.TX
    session: Session = Session.DAY


@dataclass(frozen=True)
class ContractSpec:
    """Futures contract specification."""
    product: ProductType
    point_value: Decimal        # TWD per index point
    tick_size: Decimal           # Minimum price movement
    margin_initial: Decimal     # Initial margin requirement
    margin_maintenance: Decimal # Maintenance margin
    fee_per_contract: Decimal   # Exchange + broker fee per side
    tax_rate: Decimal           # Transaction tax rate

    @property
    def tick_value(self) -> Decimal:
        """TWD value of one tick movement."""
        return self.point_value * self.tick_size


# Pre-defined contract specs
TX_SPEC = ContractSpec(
    product=ProductType.TX,
    point_value=Decimal("200"),
    tick_size=Decimal("1"),
    margin_initial=Decimal("184000"),
    margin_maintenance=Decimal("141000"),
    fee_per_contract=Decimal("60"),
    tax_rate=Decimal("0.00002"),
)

MTX_SPEC = ContractSpec(
    product=ProductType.MTX,
    point_value=Decimal("50"),
    tick_size=Decimal("1"),
    margin_initial=Decimal("46000"),
    margin_maintenance=Decimal("35250"),
    fee_per_contract=Decimal("30"),
    tax_rate=Decimal("0.00002"),
)

XMT_SPEC = ContractSpec(
    product=ProductType.XMT,
    point_value=Decimal("10"),
    tick_size=Decimal("1"),
    margin_initial=Decimal("9200"),
    margin_maintenance=Decimal("7050"),
    fee_per_contract=Decimal("15"),
    tax_rate=Decimal("0.00002"),
)

CONTRACT_SPECS: dict[ProductType, ContractSpec] = {
    ProductType.TX: TX_SPEC,
    ProductType.MTX: MTX_SPEC,
    ProductType.XMT: XMT_SPEC,
}


@dataclass(frozen=True)
class Order:
    """An order submitted to the matching engine."""
    id: UUID
    timestamp: datetime
    side: Side
    quantity: int
    order_type: OrderType
    product: ProductType = ProductType.TX
    price: Optional[Decimal] = None      # Required for LIMIT/STOP_LIMIT
    stop_price: Optional[Decimal] = None # Required for STOP/STOP_LIMIT
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[Decimal] = None
    tag: str = ""


@dataclass(frozen=True)
class Fill:
    """A fill (execution) event."""
    order_id: UUID
    timestamp: datetime
    side: Side
    price: Decimal
    quantity: int
    product: ProductType = ProductType.TX
    commission: Decimal = Decimal("0")
    tax: Decimal = Decimal("0")


@dataclass(frozen=True)
class Position:
    """Current position state."""
    product: ProductType
    side: Optional[Side]
    quantity: int
    avg_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass(frozen=True)
class Trade:
    """A completed round-trip trade."""
    id: UUID
    product: ProductType
    side: Side
    entry_time: datetime
    entry_price: Decimal
    exit_time: datetime
    exit_price: Decimal
    quantity: int
    pnl: Decimal
    commission: Decimal
    tax: Decimal
    tag: str = ""

    @property
    def net_pnl(self) -> Decimal:
        return self.pnl - self.commission - self.tax

    @property
    def holding_time(self):
        return self.exit_time - self.entry_time

    @property
    def points(self) -> Decimal:
        if self.side == Side.BUY:
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price
