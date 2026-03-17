"""Position tracking with PnL calculation."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from ..domain.enums import ProductType, Side
from ..domain.models import CONTRACT_SPECS, Fill, Position, Trade


class PositionTracker:
    """Tracks positions and generates completed trades."""

    def __init__(self):
        self._positions: dict[ProductType, Position] = {}
        self._trades: list[Trade] = []
        self._entry_fills: dict[ProductType, list[Fill]] = {}

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    def get_position(self, product: ProductType) -> Position:
        """Get current position for a product."""
        return self._positions.get(
            product,
            Position(
                product=product,
                side=None,
                quantity=0,
                avg_price=Decimal("0"),
            ),
        )

    def process_fill(self, fill: Fill) -> Optional[Trade]:
        """Process a fill and update position. Returns Trade if position closed."""
        pos = self.get_position(fill.product)
        trade: Optional[Trade] = None

        if pos.is_flat:
            # Opening new position
            new_pos = Position(
                product=fill.product,
                side=fill.side,
                quantity=fill.quantity,
                avg_price=fill.price,
            )
            self._positions[fill.product] = new_pos
            self._entry_fills.setdefault(fill.product, []).append(fill)

        elif pos.side == fill.side:
            # Adding to position
            total_qty = pos.quantity + fill.quantity
            avg = (pos.avg_price * pos.quantity + fill.price * fill.quantity) / total_qty
            new_pos = Position(
                product=fill.product,
                side=pos.side,
                quantity=total_qty,
                avg_price=avg,
            )
            self._positions[fill.product] = new_pos
            self._entry_fills.setdefault(fill.product, []).append(fill)

        else:
            # Closing/reversing position
            close_qty = min(pos.quantity, fill.quantity)
            spec = CONTRACT_SPECS[fill.product]

            if pos.side == Side.BUY:
                points = fill.price - pos.avg_price
            else:
                points = pos.avg_price - fill.price

            pnl = points * spec.point_value * close_qty

            # Accumulate commissions from entry fills
            entry_fills = self._entry_fills.get(fill.product, [])
            entry_commission = sum(f.commission for f in entry_fills)
            entry_tax = sum(f.tax for f in entry_fills)

            trade = Trade(
                id=uuid4(),
                product=fill.product,
                side=pos.side if pos.side is not None else Side.BUY,
                entry_time=entry_fills[0].timestamp if entry_fills else fill.timestamp,
                entry_price=pos.avg_price,
                exit_time=fill.timestamp,
                exit_price=fill.price,
                quantity=close_qty,
                pnl=pnl,
                commission=entry_commission + fill.commission,
                tax=entry_tax + fill.tax,
            )
            self._trades.append(trade)

            remaining = pos.quantity - close_qty
            if remaining > 0:
                new_pos = Position(
                    product=fill.product,
                    side=pos.side,
                    quantity=remaining,
                    avg_price=pos.avg_price,
                )
            elif fill.quantity > close_qty:
                # Reversal: new position in opposite direction
                new_pos = Position(
                    product=fill.product,
                    side=fill.side,
                    quantity=fill.quantity - close_qty,
                    avg_price=fill.price,
                )
                self._entry_fills[fill.product] = [fill]
            else:
                new_pos = Position(
                    product=fill.product,
                    side=None,
                    quantity=0,
                    avg_price=Decimal("0"),
                )
                self._entry_fills[fill.product] = []

            self._positions[fill.product] = new_pos

        return trade

    def update_unrealized_pnl(self, product: ProductType, current_price: Decimal) -> Position:
        """Update unrealized PnL for a position."""
        pos = self.get_position(product)
        if pos.is_flat:
            return pos

        spec = CONTRACT_SPECS[product]
        if pos.side == Side.BUY:
            points = current_price - pos.avg_price
        else:
            points = pos.avg_price - current_price

        unrealized = points * spec.point_value * pos.quantity
        updated = Position(
            product=pos.product,
            side=pos.side,
            quantity=pos.quantity,
            avg_price=pos.avg_price,
            unrealized_pnl=unrealized,
            realized_pnl=pos.realized_pnl,
        )
        self._positions[product] = updated
        return updated

    def reset(self) -> None:
        """Reset all positions and trades."""
        self._positions.clear()
        self._trades.clear()
        self._entry_fills.clear()
