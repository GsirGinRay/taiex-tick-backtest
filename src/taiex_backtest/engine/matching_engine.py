"""Matching engine for order execution simulation."""

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from ..domain.enums import OrderStatus, OrderType, Side, ProductType
from ..domain.errors import InvalidOrderError
from ..domain.models import Fill, Order
from .commission import CommissionCalculator
from .slippage import NoSlippage, SlippageModel


class MatchingEngine:
    """Simulates order matching against tick data."""

    def __init__(
        self,
        commission_calc: CommissionCalculator | None = None,
        slippage_model: SlippageModel | None = None,
    ):
        self._commission_calc = commission_calc or CommissionCalculator()
        self._slippage_model = slippage_model or NoSlippage()
        self._pending_orders: list[Order] = []

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders)

    def submit_order(self, order: Order) -> Order:
        """Submit an order to the matching engine."""
        self._validate_order(order)
        updated = Order(
            id=order.id,
            timestamp=order.timestamp,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            product=order.product,
            price=order.price,
            stop_price=order.stop_price,
            status=OrderStatus.SUBMITTED,
            filled_quantity=order.filled_quantity,
            filled_price=order.filled_price,
            tag=order.tag,
        )
        self._pending_orders.append(updated)
        return updated

    def process_tick(
        self,
        price: Decimal,
        timestamp: datetime,
        product: ProductType = ProductType.TX,
    ) -> list[Fill]:
        """Process a tick and generate fills for matching orders."""
        fills: list[Fill] = []
        remaining: list[Order] = []

        for order in self._pending_orders:
            if order.product != product:
                remaining.append(order)
                continue

            fill = self._try_fill(order, price, timestamp)
            if fill is not None:
                fills.append(fill)
            else:
                remaining.append(order)

        self._pending_orders = remaining
        return fills

    def cancel_order(self, order_id) -> Order | None:
        """Cancel a pending order by ID."""
        for i, order in enumerate(self._pending_orders):
            if order.id == order_id:
                cancelled = Order(
                    id=order.id,
                    timestamp=order.timestamp,
                    side=order.side,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    product=order.product,
                    price=order.price,
                    stop_price=order.stop_price,
                    status=OrderStatus.CANCELLED,
                    filled_quantity=order.filled_quantity,
                    filled_price=order.filled_price,
                    tag=order.tag,
                )
                self._pending_orders = [
                    o for j, o in enumerate(self._pending_orders) if j != i
                ]
                return cancelled
        return None

    def cancel_all(self) -> list[Order]:
        """Cancel all pending orders."""
        cancelled = []
        for order in self._pending_orders:
            cancelled.append(Order(
                id=order.id,
                timestamp=order.timestamp,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                product=order.product,
                price=order.price,
                stop_price=order.stop_price,
                status=OrderStatus.CANCELLED,
                filled_quantity=order.filled_quantity,
                filled_price=order.filled_price,
                tag=order.tag,
            ))
        self._pending_orders = []
        return cancelled

    def _try_fill(
        self,
        order: Order,
        tick_price: Decimal,
        timestamp: datetime,
    ) -> Fill | None:
        """Attempt to fill an order at the current tick price."""
        fill_price: Decimal | None = None

        if order.order_type == OrderType.MARKET:
            fill_price = tick_price

        elif order.order_type == OrderType.LIMIT:
            if order.price is None:
                return None
            if order.side == Side.BUY and tick_price <= order.price:
                fill_price = tick_price
            elif order.side == Side.SELL and tick_price >= order.price:
                fill_price = tick_price

        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                return None
            if order.side == Side.BUY and tick_price >= order.stop_price:
                fill_price = tick_price
            elif order.side == Side.SELL and tick_price <= order.stop_price:
                fill_price = tick_price

        if fill_price is None:
            return None

        # Apply slippage
        fill_price = self._slippage_model.calculate_slippage(
            fill_price, order.side, order.quantity
        )

        commission = self._commission_calc.calculate_commission(
            order.product, order.quantity
        )
        tax = self._commission_calc.calculate_tax(
            order.product, fill_price, order.quantity
        )

        return Fill(
            order_id=order.id,
            timestamp=timestamp,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            product=order.product,
            commission=commission,
            tax=tax,
        )

    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if order.quantity <= 0:
            raise InvalidOrderError(f"Order quantity must be positive: {order.quantity}")

        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if order.price is None:
                raise InvalidOrderError("Limit orders require a price")

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if order.stop_price is None:
                raise InvalidOrderError("Stop orders require a stop price")
