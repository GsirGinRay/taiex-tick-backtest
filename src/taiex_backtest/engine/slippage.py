"""Slippage models for realistic order execution simulation."""

from abc import ABC, abstractmethod
from decimal import Decimal

from ..domain.enums import Side


class SlippageModel(ABC):
    """Abstract base for slippage models."""

    @abstractmethod
    def calculate_slippage(
        self,
        price: Decimal,
        side: Side,
        quantity: int,
    ) -> Decimal:
        """Calculate slippage-adjusted execution price.

        Args:
            price: The current market price.
            side: Order side (BUY/SELL).
            quantity: Order quantity.

        Returns:
            The adjusted execution price after slippage.
        """


class NoSlippage(SlippageModel):
    """No slippage - execute at exact market price."""

    def calculate_slippage(self, price: Decimal, side: Side, quantity: int) -> Decimal:
        return price


class FixedSlippage(SlippageModel):
    """Fixed point slippage per trade.

    Buys execute at price + slippage_points.
    Sells execute at price - slippage_points.
    """

    def __init__(self, points: int = 1):
        if points < 0:
            raise ValueError(f"Slippage points must be non-negative: {points}")
        self._points = Decimal(str(points))

    @property
    def points(self) -> Decimal:
        return self._points

    def calculate_slippage(self, price: Decimal, side: Side, quantity: int) -> Decimal:
        if side == Side.BUY:
            return price + self._points
        return price - self._points


class PercentageSlippage(SlippageModel):
    """Percentage-based slippage.

    Buys execute at price * (1 + pct).
    Sells execute at price * (1 - pct).
    Price is rounded to nearest integer (TAIEX tick size = 1).
    """

    def __init__(self, percentage: float = 0.0001):
        if percentage < 0:
            raise ValueError(f"Slippage percentage must be non-negative: {percentage}")
        self._pct = Decimal(str(percentage))

    @property
    def percentage(self) -> Decimal:
        return self._pct

    def calculate_slippage(self, price: Decimal, side: Side, quantity: int) -> Decimal:
        slippage = (price * self._pct).quantize(Decimal("1"))
        if slippage < 1 and self._pct > 0:
            slippage = Decimal("1")
        if side == Side.BUY:
            return price + slippage
        return price - slippage


class VolumeBasedSlippage(SlippageModel):
    """Volume-dependent slippage - larger orders get more slippage.

    slippage_points = base_points + (quantity - 1) * per_contract_points
    """

    def __init__(self, base_points: int = 1, per_contract_points: float = 0.5):
        self._base = Decimal(str(base_points))
        self._per_contract = Decimal(str(per_contract_points))

    def calculate_slippage(self, price: Decimal, side: Side, quantity: int) -> Decimal:
        slippage = self._base + self._per_contract * (quantity - 1)
        slippage = slippage.quantize(Decimal("1"))
        if side == Side.BUY:
            return price + slippage
        return price - slippage
