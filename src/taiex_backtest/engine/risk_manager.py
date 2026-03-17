"""Risk management for the backtesting engine."""

from dataclasses import dataclass
from decimal import Decimal

from ..domain.enums import ProductType, Side
from ..domain.errors import InsufficientMarginError
from ..domain.models import CONTRACT_SPECS, Order, Position


@dataclass(frozen=True)
class RiskLimits:
    """Risk limit configuration."""
    max_position_size: int = 10
    max_order_size: int = 5
    max_daily_loss: Decimal = Decimal("100000")
    max_drawdown_pct: float = 0.10
    require_margin_check: bool = True


class RiskManager:
    """Manages risk limits and margin requirements."""

    def __init__(self, limits: RiskLimits | None = None):
        self._limits = limits or RiskLimits()
        self._daily_pnl = Decimal("0")
        self._peak_capital = Decimal("0")

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def daily_pnl(self) -> Decimal:
        return self._daily_pnl

    def check_order(
        self,
        order: Order,
        position: Position,
        capital: Decimal,
    ) -> list[str]:
        """Check if an order passes all risk checks.

        Returns:
            List of violation messages. Empty list means order is OK.
        """
        violations: list[str] = []

        # Check order size
        if order.quantity > self._limits.max_order_size:
            violations.append(
                f"Order size {order.quantity} exceeds max {self._limits.max_order_size}"
            )

        # Check resulting position size
        new_size = self._calculate_new_position_size(order, position)
        if new_size > self._limits.max_position_size:
            violations.append(
                f"Position size {new_size} would exceed max {self._limits.max_position_size}"
            )

        # Check daily loss limit
        if self._daily_pnl < -self._limits.max_daily_loss:
            violations.append(
                f"Daily loss {self._daily_pnl} exceeds max {self._limits.max_daily_loss}"
            )

        # Check margin requirement
        if self._limits.require_margin_check:
            margin_violations = self._check_margin(order, position, capital)
            violations.extend(margin_violations)

        return violations

    def update_pnl(self, pnl: Decimal) -> None:
        """Update daily PnL tracking."""
        self._daily_pnl += pnl

    def update_peak_capital(self, capital: Decimal) -> None:
        """Update peak capital for drawdown tracking."""
        if capital > self._peak_capital:
            self._peak_capital = capital

    def check_drawdown(self, capital: Decimal) -> bool:
        """Check if drawdown exceeds limit. Returns True if within limits."""
        if self._peak_capital <= 0:
            return True
        dd_pct = float((self._peak_capital - capital) / self._peak_capital)
        return dd_pct <= self._limits.max_drawdown_pct

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of new trading day)."""
        self._daily_pnl = Decimal("0")

    def _calculate_new_position_size(self, order: Order, position: Position) -> int:
        """Calculate the resulting position size after an order."""
        if position.is_flat:
            return order.quantity

        if position.side == order.side:
            return position.quantity + order.quantity

        # Opposite side: reducing or reversing
        remaining = position.quantity - order.quantity
        if remaining < 0:
            return abs(remaining)
        return remaining

    def _check_margin(
        self,
        order: Order,
        position: Position,
        capital: Decimal,
    ) -> list[str]:
        """Check margin requirements."""
        violations: list[str] = []

        spec = CONTRACT_SPECS.get(order.product)
        if spec is None:
            return violations

        # Only check margin for orders that increase position
        if not position.is_flat and position.side != order.side:
            return violations  # Reducing position doesn't need margin

        required_margin = spec.margin_initial * order.quantity
        if capital < required_margin:
            violations.append(
                f"Insufficient margin: need {required_margin}, have {capital}"
            )

        return violations
