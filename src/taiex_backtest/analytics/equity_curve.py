"""Equity curve calculation and analysis."""

from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

from ..domain.models import Trade


@dataclass(frozen=True)
class EquityPoint:
    """A single point on the equity curve."""
    timestamp: datetime
    equity: Decimal
    drawdown: Decimal
    drawdown_pct: float
    trade_pnl: Decimal


def build_equity_curve(
    trades: list[Trade],
    initial_capital: Decimal = Decimal("1000000"),
) -> list[EquityPoint]:
    """Build equity curve from a list of completed trades.

    Returns a list of EquityPoint objects, one per trade,
    tracking cumulative equity, drawdown, and individual trade PnL.
    """
    if not trades:
        return []

    points: list[EquityPoint] = []
    equity = initial_capital
    peak = initial_capital

    for trade in trades:
        equity += trade.net_pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        dd_pct = float(dd / peak) if peak > 0 else 0.0

        points.append(EquityPoint(
            timestamp=trade.exit_time,
            equity=equity,
            drawdown=dd,
            drawdown_pct=dd_pct,
            trade_pnl=trade.net_pnl,
        ))

    return points


def get_equity_values(
    trades: list[Trade],
    initial_capital: Decimal = Decimal("1000000"),
) -> list[Decimal]:
    """Get just the equity values list (for metrics calculation)."""
    if not trades:
        return []

    values: list[Decimal] = []
    equity = initial_capital
    for trade in trades:
        equity += trade.net_pnl
        values.append(equity)
    return values


def get_underwater_curve(points: list[EquityPoint]) -> list[tuple[datetime, float]]:
    """Get the underwater (drawdown percentage) curve."""
    return [(p.timestamp, p.drawdown_pct) for p in points]


def get_monthly_returns(
    trades: list[Trade],
    initial_capital: Decimal = Decimal("1000000"),
) -> dict[str, Decimal]:
    """Calculate monthly returns from trades.

    Returns a dict mapping 'YYYY-MM' to net PnL for that month.
    """
    monthly: dict[str, Decimal] = {}
    for trade in trades:
        key = trade.exit_time.strftime("%Y-%m")
        monthly[key] = monthly.get(key, Decimal("0")) + trade.net_pnl
    return monthly


def equity_curve_to_dicts(points: list[EquityPoint]) -> list[dict]:
    """Convert equity curve to serializable list of dicts."""
    return [
        {
            "timestamp": p.timestamp.isoformat(),
            "equity": str(p.equity),
            "drawdown": str(p.drawdown),
            "drawdown_pct": round(p.drawdown_pct, 6),
            "trade_pnl": str(p.trade_pnl),
        }
        for p in points
    ]
