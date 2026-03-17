"""Tests for Phase 4: advanced matching & risk management.

Covers slippage models, latency models, risk manager, and order manager.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from taiex_backtest.domain.enums import (
    OrderStatus,
    OrderType,
    ProductType,
    Side,
)
from taiex_backtest.domain.errors import InvalidOrderError
from taiex_backtest.domain.models import (
    CONTRACT_SPECS,
    Order,
    Position,
)
from taiex_backtest.engine.latency import (
    FixedLatency,
    LatencyModel,
    NoLatency,
    RandomLatency,
    apply_latency,
)
from taiex_backtest.engine.order_manager import OrderManager
from taiex_backtest.engine.risk_manager import RiskLimits, RiskManager
from taiex_backtest.engine.slippage import (
    FixedSlippage,
    NoSlippage,
    PercentageSlippage,
    SlippageModel,
    VolumeBasedSlippage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_position(product: ProductType = ProductType.TX) -> Position:
    """Create a flat position for testing."""
    return Position(
        product=product,
        side=None,
        quantity=0,
        avg_price=Decimal("0"),
    )


def _long_position(
    qty: int = 1,
    price: str = "20000",
    product: ProductType = ProductType.TX,
) -> Position:
    """Create a long position for testing."""
    return Position(
        product=product,
        side=Side.BUY,
        quantity=qty,
        avg_price=Decimal(price),
    )


def _short_position(
    qty: int = 1,
    price: str = "20000",
    product: ProductType = ProductType.TX,
) -> Position:
    """Create a short position for testing."""
    return Position(
        product=product,
        side=Side.SELL,
        quantity=qty,
        avg_price=Decimal(price),
    )


def _make_order(
    side: Side = Side.BUY,
    quantity: int = 1,
    order_type: OrderType = OrderType.MARKET,
    product: ProductType = ProductType.TX,
    price: Decimal | None = None,
    stop_price: Decimal | None = None,
    tag: str = "",
) -> Order:
    """Create an Order for testing."""
    return Order(
        id=uuid4(),
        timestamp=datetime(2024, 6, 1, 9, 0, 0),
        side=side,
        quantity=quantity,
        order_type=order_type,
        product=product,
        price=price,
        stop_price=stop_price,
        tag=tag,
    )


# ===================================================================
# Slippage Models
# ===================================================================

class TestSlippageModelABC:
    """Verify that SlippageModel is abstract and cannot be instantiated."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SlippageModel()  # type: ignore[abstract]


class TestNoSlippage:
    """Tests for NoSlippage model."""

    def test_buy_returns_exact_price(self):
        model = NoSlippage()
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20000")

    def test_sell_returns_exact_price(self):
        model = NoSlippage()
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 1)
        assert result == Decimal("20000")

    def test_quantity_has_no_effect(self):
        model = NoSlippage()
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 100)
        assert result == Decimal("20000")

    def test_zero_price(self):
        model = NoSlippage()
        result = model.calculate_slippage(Decimal("0"), Side.BUY, 1)
        assert result == Decimal("0")

    def test_large_price(self):
        model = NoSlippage()
        result = model.calculate_slippage(Decimal("99999"), Side.SELL, 50)
        assert result == Decimal("99999")


class TestFixedSlippage:
    """Tests for FixedSlippage model."""

    def test_default_points(self):
        model = FixedSlippage()
        assert model.points == Decimal("1")

    def test_custom_points(self):
        model = FixedSlippage(points=3)
        assert model.points == Decimal("3")

    def test_buy_adds_points(self):
        model = FixedSlippage(points=2)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20002")

    def test_sell_subtracts_points(self):
        model = FixedSlippage(points=2)
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 1)
        assert result == Decimal("19998")

    def test_zero_points(self):
        model = FixedSlippage(points=0)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20000")

    def test_negative_points_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FixedSlippage(points=-1)

    def test_quantity_has_no_effect(self):
        model = FixedSlippage(points=1)
        result_1 = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        result_10 = model.calculate_slippage(Decimal("20000"), Side.BUY, 10)
        assert result_1 == result_10

    def test_points_property_type(self):
        model = FixedSlippage(points=5)
        assert isinstance(model.points, Decimal)

    def test_buy_large_slippage(self):
        model = FixedSlippage(points=100)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20100")

    def test_sell_large_slippage(self):
        model = FixedSlippage(points=100)
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 1)
        assert result == Decimal("19900")


class TestPercentageSlippage:
    """Tests for PercentageSlippage model."""

    def test_default_percentage(self):
        model = PercentageSlippage()
        assert model.percentage == Decimal("0.0001")

    def test_custom_percentage(self):
        model = PercentageSlippage(percentage=0.001)
        assert model.percentage == Decimal("0.001")

    def test_buy_increases_price(self):
        model = PercentageSlippage(percentage=0.001)
        # 20000 * 0.001 = 20, rounded = 20
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20020")

    def test_sell_decreases_price(self):
        model = PercentageSlippage(percentage=0.001)
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 1)
        assert result == Decimal("19980")

    def test_rounding_to_integer(self):
        # 20000 * 0.0001 = 2.0 -> rounds to 2
        model = PercentageSlippage(percentage=0.0001)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20002")

    def test_minimum_1_point_when_percentage_positive(self):
        # Very small percentage on small price might yield < 1
        # e.g. 100 * 0.0001 = 0.01 -> quantize to 0 -> forced to 1
        model = PercentageSlippage(percentage=0.0001)
        result = model.calculate_slippage(Decimal("100"), Side.BUY, 1)
        assert result == Decimal("101")

    def test_zero_percentage(self):
        model = PercentageSlippage(percentage=0)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        # 20000 * 0 = 0, and pct is not > 0 so no min-1 enforcement
        assert result == Decimal("20000")

    def test_negative_percentage_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            PercentageSlippage(percentage=-0.001)

    def test_percentage_property_type(self):
        model = PercentageSlippage(percentage=0.05)
        assert isinstance(model.percentage, Decimal)

    def test_quantity_has_no_effect(self):
        model = PercentageSlippage(percentage=0.001)
        result_1 = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        result_50 = model.calculate_slippage(Decimal("20000"), Side.BUY, 50)
        assert result_1 == result_50

    def test_large_percentage_buy(self):
        model = PercentageSlippage(percentage=0.01)
        # 20000 * 0.01 = 200
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20200")

    def test_sell_with_small_price_gets_min_slippage(self):
        model = PercentageSlippage(percentage=0.0001)
        # 50 * 0.0001 = 0.005 -> quantize to 0 -> forced to 1
        result = model.calculate_slippage(Decimal("50"), Side.SELL, 1)
        assert result == Decimal("49")


class TestVolumeBasedSlippage:
    """Tests for VolumeBasedSlippage model."""

    def test_single_contract(self):
        model = VolumeBasedSlippage(base_points=1, per_contract_points=0.5)
        # slippage = 1 + (1-1)*0.5 = 1, rounded = 1
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        assert result == Decimal("20001")

    def test_multiple_contracts_buy(self):
        model = VolumeBasedSlippage(base_points=1, per_contract_points=0.5)
        # slippage = 1 + (5-1)*0.5 = 3.0, rounded = 3
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 5)
        assert result == Decimal("20003")

    def test_multiple_contracts_sell(self):
        model = VolumeBasedSlippage(base_points=1, per_contract_points=0.5)
        # slippage = 1 + (5-1)*0.5 = 3.0, rounded = 3
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 5)
        assert result == Decimal("19997")

    def test_fractional_slippage_rounds(self):
        model = VolumeBasedSlippage(base_points=1, per_contract_points=0.5)
        # slippage = 1 + (2-1)*0.5 = 1.5, rounded = 2 (banker's rounding)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 2)
        assert result == Decimal("20002")

    def test_large_quantity(self):
        model = VolumeBasedSlippage(base_points=2, per_contract_points=1)
        # slippage = 2 + (10-1)*1 = 11
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 10)
        assert result == Decimal("20011")

    def test_zero_base_points(self):
        model = VolumeBasedSlippage(base_points=0, per_contract_points=1)
        # slippage = 0 + (3-1)*1 = 2
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 3)
        assert result == Decimal("20002")

    def test_zero_per_contract(self):
        model = VolumeBasedSlippage(base_points=3, per_contract_points=0)
        # slippage = 3 + (5-1)*0 = 3
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 5)
        assert result == Decimal("20003")

    def test_sell_direction(self):
        model = VolumeBasedSlippage(base_points=2, per_contract_points=1)
        # slippage = 2 + (3-1)*1 = 4
        result = model.calculate_slippage(Decimal("20000"), Side.SELL, 3)
        assert result == Decimal("19996")

    def test_scales_with_quantity(self):
        model = VolumeBasedSlippage(base_points=1, per_contract_points=1)
        result_1 = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        result_5 = model.calculate_slippage(Decimal("20000"), Side.BUY, 5)
        # 1-contract: slippage = 1, 5-contract: slippage = 5
        assert result_5 - result_1 == Decimal("4")


# ===================================================================
# Latency Models
# ===================================================================

class TestLatencyModelABC:
    """Verify that LatencyModel is abstract."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            LatencyModel()  # type: ignore[abstract]


class TestNoLatency:
    """Tests for NoLatency model."""

    def test_returns_zero_delay(self):
        model = NoLatency()
        assert model.get_delay() == timedelta(0)

    def test_returns_timedelta(self):
        model = NoLatency()
        result = model.get_delay()
        assert isinstance(result, timedelta)

    def test_multiple_calls_consistent(self):
        model = NoLatency()
        assert model.get_delay() == model.get_delay()


class TestFixedLatency:
    """Tests for FixedLatency model."""

    def test_default_delay(self):
        model = FixedLatency()
        assert model.delay_ms == 50.0

    def test_custom_delay(self):
        model = FixedLatency(delay_ms=100.0)
        assert model.delay_ms == 100.0

    def test_get_delay_returns_correct_timedelta(self):
        model = FixedLatency(delay_ms=50.0)
        expected = timedelta(milliseconds=50.0)
        assert model.get_delay() == expected

    def test_zero_delay(self):
        model = FixedLatency(delay_ms=0)
        assert model.get_delay() == timedelta(0)
        assert model.delay_ms == 0.0

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FixedLatency(delay_ms=-1.0)

    def test_large_delay(self):
        model = FixedLatency(delay_ms=5000.0)
        assert model.get_delay() == timedelta(milliseconds=5000)

    def test_fractional_delay(self):
        model = FixedLatency(delay_ms=0.5)
        assert model.delay_ms == 0.5
        assert model.get_delay() == timedelta(milliseconds=0.5)

    def test_multiple_calls_consistent(self):
        model = FixedLatency(delay_ms=75.0)
        assert model.get_delay() == model.get_delay()

    def test_delay_ms_property_type(self):
        model = FixedLatency(delay_ms=50.0)
        assert isinstance(model.delay_ms, float)


class TestRandomLatency:
    """Tests for RandomLatency model."""

    def test_returns_timedelta(self):
        model = RandomLatency(seed=42)
        result = model.get_delay()
        assert isinstance(result, timedelta)

    def test_clamped_to_min(self):
        # Use a very low mean and high std to occasionally generate below min
        model = RandomLatency(
            mean_ms=10.0, std_ms=100.0, min_ms=5.0, max_ms=500.0, seed=42,
        )
        for _ in range(100):
            delay = model.get_delay()
            assert delay >= timedelta(milliseconds=5.0)

    def test_clamped_to_max(self):
        model = RandomLatency(
            mean_ms=400.0, std_ms=200.0, min_ms=1.0, max_ms=500.0, seed=42,
        )
        for _ in range(100):
            delay = model.get_delay()
            assert delay <= timedelta(milliseconds=500.0)

    def test_deterministic_with_seed(self):
        model_a = RandomLatency(seed=123)
        model_b = RandomLatency(seed=123)
        delays_a = [model_a.get_delay() for _ in range(10)]
        delays_b = [model_b.get_delay() for _ in range(10)]
        assert delays_a == delays_b

    def test_different_seeds_differ(self):
        model_a = RandomLatency(seed=1)
        model_b = RandomLatency(seed=2)
        delays_a = [model_a.get_delay() for _ in range(10)]
        delays_b = [model_b.get_delay() for _ in range(10)]
        assert delays_a != delays_b

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RandomLatency(mean_ms=-10.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RandomLatency(std_ms=-5.0)

    def test_zero_std_gives_approximately_mean(self):
        model = RandomLatency(
            mean_ms=50.0, std_ms=0.0, min_ms=1.0, max_ms=100.0, seed=42,
        )
        delay = model.get_delay()
        assert delay == timedelta(milliseconds=50.0)

    def test_default_parameters(self):
        model = RandomLatency(seed=42)
        delay = model.get_delay()
        # Default min=10, max=500 so delay should be within range
        assert timedelta(milliseconds=10.0) <= delay <= timedelta(milliseconds=500.0)

    def test_many_samples_within_bounds(self):
        model = RandomLatency(
            mean_ms=50.0, std_ms=20.0, min_ms=10.0, max_ms=200.0, seed=42,
        )
        for _ in range(500):
            delay = model.get_delay()
            delay_ms = delay.total_seconds() * 1000
            assert 10.0 <= delay_ms <= 200.0


class TestApplyLatency:
    """Tests for the apply_latency function."""

    def test_with_no_latency(self):
        ts = datetime(2024, 6, 1, 9, 0, 0)
        result = apply_latency(ts, NoLatency())
        assert result == ts

    def test_with_fixed_latency(self):
        ts = datetime(2024, 6, 1, 9, 0, 0)
        result = apply_latency(ts, FixedLatency(delay_ms=100.0))
        expected = ts + timedelta(milliseconds=100.0)
        assert result == expected

    def test_result_is_after_original(self):
        ts = datetime(2024, 6, 1, 9, 0, 0)
        result = apply_latency(ts, FixedLatency(delay_ms=50.0))
        assert result > ts

    def test_with_random_latency(self):
        ts = datetime(2024, 6, 1, 9, 0, 0)
        model = RandomLatency(
            mean_ms=50.0, std_ms=10.0, min_ms=10.0, max_ms=200.0, seed=42,
        )
        result = apply_latency(ts, model)
        assert result > ts
        delay_ms = (result - ts).total_seconds() * 1000
        assert 10.0 <= delay_ms <= 200.0

    def test_preserves_microseconds(self):
        ts = datetime(2024, 6, 1, 9, 0, 0, 123456)
        result = apply_latency(ts, FixedLatency(delay_ms=0))
        assert result == ts


# ===================================================================
# Risk Manager
# ===================================================================

class TestRiskLimits:
    """Tests for RiskLimits dataclass."""

    def test_default_values(self):
        limits = RiskLimits()
        assert limits.max_position_size == 10
        assert limits.max_order_size == 5
        assert limits.max_daily_loss == Decimal("100000")
        assert limits.max_drawdown_pct == 0.10
        assert limits.require_margin_check is True

    def test_custom_values(self):
        limits = RiskLimits(
            max_position_size=20,
            max_order_size=10,
            max_daily_loss=Decimal("200000"),
            max_drawdown_pct=0.15,
            require_margin_check=False,
        )
        assert limits.max_position_size == 20
        assert limits.max_order_size == 10
        assert limits.max_daily_loss == Decimal("200000")
        assert limits.max_drawdown_pct == 0.15
        assert limits.require_margin_check is False

    def test_frozen(self):
        limits = RiskLimits()
        with pytest.raises(AttributeError):
            limits.max_position_size = 99  # type: ignore[misc]


class TestRiskManagerOrderChecks:
    """Tests for RiskManager.check_order."""

    def test_valid_order_returns_empty(self):
        rm = RiskManager(RiskLimits(require_margin_check=False))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert violations == []

    def test_order_size_exceeds_max(self):
        rm = RiskManager(RiskLimits(max_order_size=3, require_margin_check=False))
        order = _make_order(side=Side.BUY, quantity=5)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert len(violations) == 1
        assert "Order size" in violations[0]
        assert "5" in violations[0]

    def test_order_size_at_exact_max_passes(self):
        rm = RiskManager(RiskLimits(max_order_size=5, require_margin_check=False))
        order = _make_order(side=Side.BUY, quantity=5)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert violations == []

    def test_position_size_exceeds_max_from_flat(self):
        rm = RiskManager(RiskLimits(
            max_position_size=3,
            max_order_size=10,
            require_margin_check=False,
        ))
        order = _make_order(side=Side.BUY, quantity=5)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert any("Position size" in v for v in violations)

    def test_position_size_exceeds_max_adding_to_existing(self):
        rm = RiskManager(RiskLimits(
            max_position_size=5,
            max_order_size=10,
            require_margin_check=False,
        ))
        order = _make_order(side=Side.BUY, quantity=3)
        position = _long_position(qty=4)
        # new_size = 4 + 3 = 7 > 5
        violations = rm.check_order(order, position, Decimal("1000000"))
        assert any("Position size" in v for v in violations)

    def test_reducing_position_within_limits(self):
        rm = RiskManager(RiskLimits(
            max_position_size=5,
            max_order_size=10,
            require_margin_check=False,
        ))
        order = _make_order(side=Side.SELL, quantity=2)
        position = _long_position(qty=5)
        # new_size = 5 - 2 = 3 <= 5
        violations = rm.check_order(order, position, Decimal("1000000"))
        assert violations == []

    def test_reversing_position_size_check(self):
        rm = RiskManager(RiskLimits(
            max_position_size=3,
            max_order_size=10,
            require_margin_check=False,
        ))
        order = _make_order(side=Side.SELL, quantity=5)
        position = _long_position(qty=1)
        # remaining = 1 - 5 = -4, new_size = abs(-4) = 4 > 3
        violations = rm.check_order(order, position, Decimal("1000000"))
        assert any("Position size" in v for v in violations)

    def test_daily_loss_violation(self):
        rm = RiskManager(RiskLimits(
            max_daily_loss=Decimal("50000"),
            require_margin_check=False,
        ))
        # Accumulate enough losses to trigger
        rm.update_pnl(Decimal("-60000"))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert any("Daily loss" in v for v in violations)

    def test_daily_loss_at_exact_limit_passes(self):
        rm = RiskManager(RiskLimits(
            max_daily_loss=Decimal("50000"),
            require_margin_check=False,
        ))
        # pnl = -50000, check is pnl < -max => -50000 < -50000 is False
        rm.update_pnl(Decimal("-50000"))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert not any("Daily loss" in v for v in violations)

    def test_multiple_violations(self):
        rm = RiskManager(RiskLimits(
            max_order_size=2,
            max_position_size=3,
            max_daily_loss=Decimal("10000"),
            require_margin_check=False,
        ))
        rm.update_pnl(Decimal("-20000"))
        order = _make_order(side=Side.BUY, quantity=5)
        violations = rm.check_order(order, _flat_position(), Decimal("1000000"))
        assert len(violations) >= 3  # order size + position size + daily loss


class TestRiskManagerMarginChecks:
    """Tests for RiskManager margin checking."""

    def test_sufficient_margin_passes(self):
        rm = RiskManager(RiskLimits(require_margin_check=True))
        order = _make_order(side=Side.BUY, quantity=1)
        # TX initial margin = 184000
        violations = rm.check_order(order, _flat_position(), Decimal("200000"))
        assert not any("margin" in v.lower() for v in violations)

    def test_insufficient_margin_fails(self):
        rm = RiskManager(RiskLimits(require_margin_check=True))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("100000"))
        assert any("margin" in v.lower() for v in violations)

    def test_multiple_contracts_margin(self):
        rm = RiskManager(RiskLimits(
            max_order_size=5,
            max_position_size=10,
            require_margin_check=True,
        ))
        order = _make_order(side=Side.BUY, quantity=3)
        # 3 * 184000 = 552000
        violations = rm.check_order(order, _flat_position(), Decimal("500000"))
        assert any("margin" in v.lower() for v in violations)

    def test_margin_exact_amount_passes(self):
        rm = RiskManager(RiskLimits(require_margin_check=True))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("184000"))
        assert not any("margin" in v.lower() for v in violations)

    def test_reducing_position_skips_margin_check(self):
        rm = RiskManager(RiskLimits(require_margin_check=True))
        order = _make_order(side=Side.SELL, quantity=1)
        position = _long_position(qty=2)
        # Reducing a long position by selling: no margin required
        violations = rm.check_order(order, position, Decimal("0"))
        assert not any("margin" in v.lower() for v in violations)

    def test_margin_disabled(self):
        rm = RiskManager(RiskLimits(require_margin_check=False))
        order = _make_order(side=Side.BUY, quantity=1)
        violations = rm.check_order(order, _flat_position(), Decimal("0"))
        assert not any("margin" in v.lower() for v in violations)

    def test_adding_to_position_checks_margin(self):
        rm = RiskManager(RiskLimits(
            max_order_size=10,
            max_position_size=10,
            require_margin_check=True,
        ))
        order = _make_order(side=Side.BUY, quantity=2)
        position = _long_position(qty=1)
        # Adding 2 to long: 2 * 184000 = 368000
        violations = rm.check_order(order, position, Decimal("300000"))
        assert any("margin" in v.lower() for v in violations)

    def test_mtx_margin_check(self):
        rm = RiskManager(RiskLimits(
            max_order_size=10,
            max_position_size=10,
            require_margin_check=True,
        ))
        order = _make_order(side=Side.BUY, quantity=1, product=ProductType.MTX)
        flat = _flat_position(product=ProductType.MTX)
        # MTX margin_initial = 46000
        violations = rm.check_order(order, flat, Decimal("40000"))
        assert any("margin" in v.lower() for v in violations)


class TestRiskManagerPnL:
    """Tests for RiskManager PnL tracking."""

    def test_initial_daily_pnl_is_zero(self):
        rm = RiskManager()
        assert rm.daily_pnl == Decimal("0")

    def test_update_pnl_positive(self):
        rm = RiskManager()
        rm.update_pnl(Decimal("5000"))
        assert rm.daily_pnl == Decimal("5000")

    def test_update_pnl_negative(self):
        rm = RiskManager()
        rm.update_pnl(Decimal("-3000"))
        assert rm.daily_pnl == Decimal("-3000")

    def test_update_pnl_cumulative(self):
        rm = RiskManager()
        rm.update_pnl(Decimal("5000"))
        rm.update_pnl(Decimal("-8000"))
        rm.update_pnl(Decimal("1000"))
        assert rm.daily_pnl == Decimal("-2000")

    def test_reset_daily(self):
        rm = RiskManager()
        rm.update_pnl(Decimal("-50000"))
        rm.reset_daily()
        assert rm.daily_pnl == Decimal("0")


class TestRiskManagerDrawdown:
    """Tests for RiskManager drawdown checking."""

    def test_drawdown_within_limits(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm.update_peak_capital(Decimal("1000000"))
        assert rm.check_drawdown(Decimal("950000")) is True

    def test_drawdown_exceeds_limits(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm.update_peak_capital(Decimal("1000000"))
        # 15% drawdown > 10% limit
        assert rm.check_drawdown(Decimal("850000")) is False

    def test_drawdown_at_exact_limit(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm.update_peak_capital(Decimal("1000000"))
        # Exactly 10%
        assert rm.check_drawdown(Decimal("900000")) is True

    def test_drawdown_no_peak_capital(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        # peak = 0, should return True (within limits)
        assert rm.check_drawdown(Decimal("500000")) is True

    def test_update_peak_capital_only_increases(self):
        rm = RiskManager()
        rm.update_peak_capital(Decimal("1000000"))
        rm.update_peak_capital(Decimal("800000"))  # Lower, should not update
        rm.update_peak_capital(Decimal("1200000"))
        # Drawdown from 1200000 to 1000000 = 16.7%
        assert rm.check_drawdown(Decimal("1000000")) is False  # 16.7% > 10%

    def test_drawdown_zero_capital(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.10))
        rm.update_peak_capital(Decimal("1000000"))
        # 100% drawdown
        assert rm.check_drawdown(Decimal("0")) is False

    def test_peak_tracks_highest_value(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.20))
        rm.update_peak_capital(Decimal("500000"))
        rm.update_peak_capital(Decimal("800000"))
        rm.update_peak_capital(Decimal("700000"))
        # Peak = 800000, 15% drawdown = 680000
        assert rm.check_drawdown(Decimal("680000")) is True
        # 25% drawdown = 600000
        assert rm.check_drawdown(Decimal("600000")) is False


class TestRiskManagerDefaults:
    """Tests for RiskManager default construction."""

    def test_default_limits(self):
        rm = RiskManager()
        assert rm.limits.max_position_size == 10
        assert rm.limits.max_order_size == 5

    def test_custom_limits(self):
        limits = RiskLimits(max_position_size=20)
        rm = RiskManager(limits)
        assert rm.limits.max_position_size == 20


# ===================================================================
# Order Manager
# ===================================================================

class TestOrderManagerCreate:
    """Tests for OrderManager.create_order."""

    def test_create_market_order(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )
        assert order.side == Side.BUY
        assert order.quantity == 1
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.PENDING
        assert order.product == ProductType.TX

    def test_create_limit_order(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.SELL,
            quantity=2,
            order_type=OrderType.LIMIT,
            price=Decimal("20100"),
        )
        assert order.order_type == OrderType.LIMIT
        assert order.price == Decimal("20100")

    def test_create_stop_order(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.STOP,
            stop_price=Decimal("20200"),
        )
        assert order.stop_price == Decimal("20200")

    def test_create_stop_limit_order(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.STOP_LIMIT,
            price=Decimal("20100"),
            stop_price=Decimal("20200"),
        )
        assert order.price == Decimal("20100")
        assert order.stop_price == Decimal("20200")

    def test_create_with_tag(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            tag="entry_signal",
        )
        assert order.tag == "entry_signal"

    def test_create_with_custom_product(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            product=ProductType.MTX,
        )
        assert order.product == ProductType.MTX

    def test_create_with_timestamp(self):
        om = OrderManager()
        ts = datetime(2024, 6, 1, 10, 30, 0)
        order = om.create_order(
            side=Side.BUY,
            quantity=1,
            timestamp=ts,
        )
        assert order.timestamp == ts

    def test_zero_quantity_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="positive"):
            om.create_order(side=Side.BUY, quantity=0)

    def test_negative_quantity_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="positive"):
            om.create_order(side=Side.BUY, quantity=-1)

    def test_limit_order_without_price_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="price"):
            om.create_order(
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
            )

    def test_stop_limit_without_price_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="price"):
            om.create_order(
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.STOP_LIMIT,
                stop_price=Decimal("20000"),
            )

    def test_stop_order_without_stop_price_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="stop price"):
            om.create_order(
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.STOP,
            )

    def test_stop_limit_without_stop_price_raises(self):
        om = OrderManager()
        with pytest.raises(InvalidOrderError, match="stop price"):
            om.create_order(
                side=Side.BUY,
                quantity=1,
                order_type=OrderType.STOP_LIMIT,
                price=Decimal("20000"),
            )

    def test_order_gets_unique_id(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        o2 = om.create_order(side=Side.SELL, quantity=1)
        assert o1.id != o2.id

    def test_order_registered_internally(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        assert om.get_order(order.id) is order


class TestOrderManagerGet:
    """Tests for OrderManager.get_order."""

    def test_get_existing_order(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        found = om.get_order(order.id)
        assert found is not None
        assert found.id == order.id

    def test_get_nonexistent_order(self):
        om = OrderManager()
        assert om.get_order(uuid4()) is None


class TestOrderManagerUpdateStatus:
    """Tests for OrderManager.update_status."""

    def test_update_to_submitted(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        updated = om.update_status(order.id, OrderStatus.SUBMITTED)
        assert updated is not None
        assert updated.status == OrderStatus.SUBMITTED
        assert updated.id == order.id

    def test_update_to_rejected(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        updated = om.update_status(order.id, OrderStatus.REJECTED)
        assert updated is not None
        assert updated.status == OrderStatus.REJECTED

    def test_update_preserves_fields(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.SELL,
            quantity=3,
            order_type=OrderType.LIMIT,
            price=Decimal("20100"),
            tag="test",
        )
        updated = om.update_status(order.id, OrderStatus.SUBMITTED)
        assert updated is not None
        assert updated.side == Side.SELL
        assert updated.quantity == 3
        assert updated.price == Decimal("20100")
        assert updated.tag == "test"

    def test_update_nonexistent_returns_none(self):
        om = OrderManager()
        assert om.update_status(uuid4(), OrderStatus.SUBMITTED) is None

    def test_updated_order_replaces_in_store(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.update_status(order.id, OrderStatus.SUBMITTED)
        current = om.get_order(order.id)
        assert current is not None
        assert current.status == OrderStatus.SUBMITTED


class TestOrderManagerFill:
    """Tests for OrderManager.mark_filled."""

    def test_full_fill(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=2)
        filled = om.mark_filled(order.id, Decimal("20050"))
        assert filled is not None
        assert filled.status == OrderStatus.FILLED
        assert filled.filled_price == Decimal("20050")
        assert filled.filled_quantity == 2

    def test_partial_fill(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=5)
        filled = om.mark_filled(order.id, Decimal("20050"), filled_quantity=3)
        assert filled is not None
        assert filled.status == OrderStatus.PARTIAL
        assert filled.filled_quantity == 3

    def test_fill_nonexistent_returns_none(self):
        om = OrderManager()
        assert om.mark_filled(uuid4(), Decimal("20000")) is None

    def test_fill_exact_quantity_is_filled_status(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=3)
        filled = om.mark_filled(order.id, Decimal("20050"), filled_quantity=3)
        assert filled is not None
        assert filled.status == OrderStatus.FILLED

    def test_fill_preserves_order_fields(self):
        om = OrderManager()
        order = om.create_order(
            side=Side.SELL,
            quantity=2,
            order_type=OrderType.LIMIT,
            price=Decimal("20100"),
            tag="exit",
        )
        filled = om.mark_filled(order.id, Decimal("20100"))
        assert filled is not None
        assert filled.side == Side.SELL
        assert filled.order_type == OrderType.LIMIT
        assert filled.price == Decimal("20100")
        assert filled.tag == "exit"

    def test_fill_updates_store(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.mark_filled(order.id, Decimal("20000"))
        current = om.get_order(order.id)
        assert current is not None
        assert current.status == OrderStatus.FILLED


class TestOrderManagerCancel:
    """Tests for OrderManager cancel operations."""

    def test_cancel_single_order(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        cancelled = om.cancel_order(order.id)
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED

    def test_cancel_nonexistent_returns_none(self):
        om = OrderManager()
        assert om.cancel_order(uuid4()) is None

    def test_cancel_all(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        om.create_order(side=Side.BUY, quantity=2)
        cancelled = om.cancel_all()
        assert len(cancelled) == 3
        for o in cancelled:
            assert o.status == OrderStatus.CANCELLED

    def test_cancel_all_empty(self):
        om = OrderManager()
        cancelled = om.cancel_all()
        assert cancelled == []

    def test_cancel_all_skips_already_filled(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        o2 = om.create_order(side=Side.SELL, quantity=1)
        om.mark_filled(o1.id, Decimal("20000"))
        cancelled = om.cancel_all()
        # Only o2 should be cancelled (o1 is FILLED, not active)
        assert len(cancelled) == 1
        assert cancelled[0].id == o2.id

    def test_cancel_all_skips_already_cancelled(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        o2 = om.create_order(side=Side.SELL, quantity=1)
        om.cancel_order(o1.id)
        cancelled = om.cancel_all()
        # Only o2 should be newly cancelled
        assert len(cancelled) == 1
        assert cancelled[0].id == o2.id


class TestOrderManagerActiveOrders:
    """Tests for OrderManager.active_orders property."""

    def test_no_orders(self):
        om = OrderManager()
        assert om.active_orders == []

    def test_pending_orders_are_active(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        assert len(om.active_orders) == 1

    def test_submitted_orders_are_active(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.update_status(order.id, OrderStatus.SUBMITTED)
        assert len(om.active_orders) == 1

    def test_filled_orders_not_active(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.mark_filled(order.id, Decimal("20000"))
        assert len(om.active_orders) == 0

    def test_cancelled_orders_not_active(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.cancel_order(order.id)
        assert len(om.active_orders) == 0

    def test_rejected_orders_not_active(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.update_status(order.id, OrderStatus.REJECTED)
        assert len(om.active_orders) == 0

    def test_mixed_statuses(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        o2 = om.create_order(side=Side.SELL, quantity=1)
        o3 = om.create_order(side=Side.BUY, quantity=1)
        om.mark_filled(o1.id, Decimal("20000"))
        om.cancel_order(o2.id)
        # Only o3 is active
        assert len(om.active_orders) == 1
        assert om.active_orders[0].id == o3.id


class TestOrderManagerHistory:
    """Tests for OrderManager.order_history and order_count."""

    def test_empty_history(self):
        om = OrderManager()
        assert om.order_history == []
        assert om.order_count == 0

    def test_history_tracks_creation(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        assert om.order_count == 2
        assert len(om.order_history) == 2

    def test_history_returns_copy(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        h1 = om.order_history
        h2 = om.order_history
        assert h1 is not h2
        assert h1 == h2

    def test_history_not_affected_by_status_change(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.mark_filled(order.id, Decimal("20000"))
        # History count stays the same (no new order created)
        assert om.order_count == 1


class TestOrderManagerQueryByTag:
    """Tests for OrderManager.get_orders_by_tag."""

    def test_get_by_tag(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1, tag="alpha")
        om.create_order(side=Side.SELL, quantity=1, tag="beta")
        om.create_order(side=Side.BUY, quantity=2, tag="alpha")
        results = om.get_orders_by_tag("alpha")
        assert len(results) == 2
        for o in results:
            assert o.tag == "alpha"

    def test_get_by_tag_empty(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1, tag="alpha")
        assert om.get_orders_by_tag("beta") == []

    def test_get_by_tag_default_empty_string(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        results = om.get_orders_by_tag("")
        assert len(results) == 1


class TestOrderManagerQueryByStatus:
    """Tests for OrderManager.get_orders_by_status."""

    def test_get_pending(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        results = om.get_orders_by_status(OrderStatus.PENDING)
        assert len(results) == 2

    def test_get_filled(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        om.mark_filled(o1.id, Decimal("20000"))
        results = om.get_orders_by_status(OrderStatus.FILLED)
        assert len(results) == 1
        assert results[0].id == o1.id

    def test_get_cancelled(self):
        om = OrderManager()
        o1 = om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        om.cancel_order(o1.id)
        results = om.get_orders_by_status(OrderStatus.CANCELLED)
        assert len(results) == 1

    def test_get_nonexistent_status(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        results = om.get_orders_by_status(OrderStatus.REJECTED)
        assert results == []


class TestOrderManagerReset:
    """Tests for OrderManager.reset."""

    def test_reset_clears_everything(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        om.create_order(side=Side.SELL, quantity=1)
        om.reset()
        assert om.active_orders == []
        assert om.order_history == []
        assert om.order_count == 0

    def test_reset_allows_new_orders(self):
        om = OrderManager()
        om.create_order(side=Side.BUY, quantity=1)
        om.reset()
        new_order = om.create_order(side=Side.SELL, quantity=1)
        assert om.order_count == 1
        assert om.get_order(new_order.id) is not None

    def test_reset_old_orders_unreachable(self):
        om = OrderManager()
        old_order = om.create_order(side=Side.BUY, quantity=1)
        om.reset()
        assert om.get_order(old_order.id) is None


# ===================================================================
# Edge cases & cross-cutting concerns
# ===================================================================

class TestSlippageEdgeCases:
    """Edge cases for slippage models."""

    def test_no_slippage_is_subclass(self):
        assert isinstance(NoSlippage(), SlippageModel)

    def test_fixed_slippage_is_subclass(self):
        assert isinstance(FixedSlippage(), SlippageModel)

    def test_percentage_slippage_is_subclass(self):
        assert isinstance(PercentageSlippage(), SlippageModel)

    def test_volume_based_slippage_is_subclass(self):
        assert isinstance(VolumeBasedSlippage(), SlippageModel)

    def test_fixed_slippage_with_very_large_price(self):
        model = FixedSlippage(points=1)
        result = model.calculate_slippage(Decimal("999999"), Side.BUY, 1)
        assert result == Decimal("1000000")

    def test_percentage_slippage_precise_rounding(self):
        # 15555 * 0.0003 = 4.6665, quantize("1") => 5 (banker's rounding)
        model = PercentageSlippage(percentage=0.0003)
        result = model.calculate_slippage(Decimal("15555"), Side.BUY, 1)
        # quantize to integer: 4.6665 rounds to 5
        assert result == Decimal("15560")

    def test_volume_slippage_single_contract_equals_base(self):
        model = VolumeBasedSlippage(base_points=5, per_contract_points=2)
        result = model.calculate_slippage(Decimal("20000"), Side.BUY, 1)
        # 5 + (1-1)*2 = 5
        assert result == Decimal("20005")


class TestLatencyEdgeCases:
    """Edge cases for latency models."""

    def test_no_latency_is_subclass(self):
        assert isinstance(NoLatency(), LatencyModel)

    def test_fixed_latency_is_subclass(self):
        assert isinstance(FixedLatency(), LatencyModel)

    def test_random_latency_is_subclass(self):
        assert isinstance(RandomLatency(seed=42), LatencyModel)

    def test_apply_latency_with_midnight_crossing(self):
        ts = datetime(2024, 6, 1, 23, 59, 59, 990000)
        result = apply_latency(ts, FixedLatency(delay_ms=20.0))
        assert result.day == 2
        assert result.hour == 0
        assert result.minute == 0

    def test_random_latency_zero_std_reproducible(self):
        model = RandomLatency(
            mean_ms=100.0, std_ms=0.0, min_ms=0.0, max_ms=200.0, seed=1,
        )
        delays = [model.get_delay() for _ in range(10)]
        assert all(d == delays[0] for d in delays)


class TestRiskManagerEdgeCases:
    """Edge cases for RiskManager."""

    def test_check_order_short_position_sell_adds(self):
        """Selling when already short should increase position size."""
        rm = RiskManager(RiskLimits(
            max_position_size=3,
            max_order_size=10,
            require_margin_check=False,
        ))
        order = _make_order(side=Side.SELL, quantity=2)
        position = _short_position(qty=2)
        # Same side: 2 + 2 = 4 > 3
        violations = rm.check_order(order, position, Decimal("1000000"))
        assert any("Position size" in v for v in violations)

    def test_multiple_pnl_updates_then_reset(self):
        rm = RiskManager()
        rm.update_pnl(Decimal("-30000"))
        rm.update_pnl(Decimal("-40000"))
        rm.reset_daily()
        rm.update_pnl(Decimal("-5000"))
        assert rm.daily_pnl == Decimal("-5000")

    def test_peak_capital_starts_at_zero(self):
        rm = RiskManager()
        # No peak set, drawdown check should pass
        assert rm.check_drawdown(Decimal("0")) is True


class TestOrderManagerEdgeCases:
    """Edge cases for OrderManager."""

    def test_create_many_orders(self):
        om = OrderManager()
        for i in range(100):
            om.create_order(side=Side.BUY, quantity=1, tag=f"order_{i}")
        assert om.order_count == 100
        assert len(om.active_orders) == 100

    def test_partial_fill_then_full_fill(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=5)
        partial = om.mark_filled(order.id, Decimal("20000"), filled_quantity=3)
        assert partial is not None
        assert partial.status == OrderStatus.PARTIAL
        full = om.mark_filled(order.id, Decimal("20010"), filled_quantity=5)
        assert full is not None
        assert full.status == OrderStatus.FILLED

    def test_cancel_after_fill_still_cancels(self):
        """OrderManager does not enforce status transitions; it just sets status."""
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        om.mark_filled(order.id, Decimal("20000"))
        # cancel_order just calls update_status to CANCELLED
        # But cancel_all filters by active_orders, which excludes FILLED
        # Direct cancel_order still works (no guard)
        cancelled = om.cancel_order(order.id)
        assert cancelled is not None
        assert cancelled.status == OrderStatus.CANCELLED

    def test_get_orders_by_tag_after_status_change(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1, tag="signal_a")
        om.mark_filled(order.id, Decimal("20000"))
        results = om.get_orders_by_tag("signal_a")
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED

    def test_immutable_order_objects(self):
        om = OrderManager()
        order = om.create_order(side=Side.BUY, quantity=1)
        with pytest.raises(AttributeError):
            order.quantity = 99  # type: ignore[misc]
