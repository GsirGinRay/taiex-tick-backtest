"""Commission and tax calculation for TAIEX futures."""

from decimal import Decimal

from ..domain.enums import ProductType
from ..domain.models import CONTRACT_SPECS, ContractSpec


class CommissionCalculator:
    """Calculates trading commissions and taxes."""

    def __init__(self, spec: ContractSpec | None = None):
        self._spec = spec

    def get_spec(self, product: ProductType) -> ContractSpec:
        """Get contract spec, using override or default."""
        if self._spec is not None:
            return self._spec
        return CONTRACT_SPECS[product]

    def calculate_commission(
        self,
        product: ProductType,
        quantity: int,
    ) -> Decimal:
        """Calculate commission for a trade (one side)."""
        spec = self.get_spec(product)
        return spec.fee_per_contract * quantity

    def calculate_tax(
        self,
        product: ProductType,
        price: Decimal,
        quantity: int,
    ) -> Decimal:
        """Calculate transaction tax for a trade (one side)."""
        spec = self.get_spec(product)
        notional = price * spec.point_value * quantity
        return (notional * spec.tax_rate).quantize(Decimal("1"))

    def calculate_total_cost(
        self,
        product: ProductType,
        price: Decimal,
        quantity: int,
    ) -> Decimal:
        """Calculate total cost (commission + tax) for one side."""
        return (
            self.calculate_commission(product, quantity)
            + self.calculate_tax(product, price, quantity)
        )
