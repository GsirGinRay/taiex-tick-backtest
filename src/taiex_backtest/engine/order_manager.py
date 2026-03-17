"""Order manager for tracking order lifecycle."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from ..domain.enums import OrderStatus, OrderType, ProductType, Side
from ..domain.errors import InvalidOrderError
from ..domain.models import Order


class OrderManager:
    """Manages order creation, tracking, and lifecycle."""

    def __init__(self):
        self._orders: dict[UUID, Order] = {}
        self._order_history: list[Order] = []

    @property
    def active_orders(self) -> list[Order]:
        """Get all active (pending/submitted) orders."""
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
        ]

    @property
    def order_history(self) -> list[Order]:
        """Get complete order history."""
        return list(self._order_history)

    @property
    def order_count(self) -> int:
        """Total number of orders created."""
        return len(self._order_history)

    def create_order(
        self,
        side: Side,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        product: ProductType = ProductType.TX,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        timestamp: datetime | None = None,
        tag: str = "",
    ) -> Order:
        """Create and register a new order."""
        if quantity <= 0:
            raise InvalidOrderError(f"Quantity must be positive: {quantity}")

        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and price is None:
            raise InvalidOrderError("Limit orders require a price")

        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and stop_price is None:
            raise InvalidOrderError("Stop orders require a stop price")

        order = Order(
            id=uuid4(),
            timestamp=timestamp or datetime.now(),
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
            stop_price=stop_price,
            tag=tag,
        )

        self._orders[order.id] = order
        self._order_history.append(order)
        return order

    def get_order(self, order_id: UUID) -> Order | None:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def update_status(self, order_id: UUID, status: OrderStatus) -> Order | None:
        """Update an order's status (creates new immutable order)."""
        old = self._orders.get(order_id)
        if old is None:
            return None

        updated = Order(
            id=old.id,
            timestamp=old.timestamp,
            side=old.side,
            quantity=old.quantity,
            order_type=old.order_type,
            product=old.product,
            price=old.price,
            stop_price=old.stop_price,
            status=status,
            filled_quantity=old.filled_quantity,
            filled_price=old.filled_price,
            tag=old.tag,
        )
        self._orders[order_id] = updated
        return updated

    def mark_filled(
        self,
        order_id: UUID,
        filled_price: Decimal,
        filled_quantity: int | None = None,
    ) -> Order | None:
        """Mark an order as filled."""
        old = self._orders.get(order_id)
        if old is None:
            return None

        qty = filled_quantity or old.quantity
        status = OrderStatus.FILLED if qty >= old.quantity else OrderStatus.PARTIAL

        updated = Order(
            id=old.id,
            timestamp=old.timestamp,
            side=old.side,
            quantity=old.quantity,
            order_type=old.order_type,
            product=old.product,
            price=old.price,
            stop_price=old.stop_price,
            status=status,
            filled_quantity=qty,
            filled_price=filled_price,
            tag=old.tag,
        )
        self._orders[order_id] = updated
        return updated

    def cancel_order(self, order_id: UUID) -> Order | None:
        """Cancel an order."""
        return self.update_status(order_id, OrderStatus.CANCELLED)

    def cancel_all(self) -> list[Order]:
        """Cancel all active orders."""
        cancelled: list[Order] = []
        for order in self.active_orders:
            result = self.cancel_order(order.id)
            if result is not None:
                cancelled.append(result)
        return cancelled

    def get_orders_by_tag(self, tag: str) -> list[Order]:
        """Get all orders with a specific tag."""
        return [o for o in self._orders.values() if o.tag == tag]

    def get_orders_by_status(self, status: OrderStatus) -> list[Order]:
        """Get all orders with a specific status."""
        return [o for o in self._orders.values() if o.status == status]

    def reset(self) -> None:
        """Reset all orders."""
        self._orders.clear()
        self._order_history.clear()
