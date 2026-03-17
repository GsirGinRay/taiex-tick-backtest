"""Trade-level analysis and breakdown."""

from dataclasses import dataclass
from decimal import Decimal
from datetime import timedelta

from ..domain.models import Trade
from ..domain.enums import Side


@dataclass(frozen=True)
class TradeStats:
    """Statistical summary of a set of trades."""
    count: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: Decimal
    avg_pnl: Decimal
    median_pnl: Decimal
    std_pnl: Decimal
    avg_holding_time: timedelta
    avg_points: Decimal


def analyze_trades(trades: list[Trade]) -> TradeStats:
    """Compute statistical summary of trades."""
    if not trades:
        return TradeStats(
            count=0, winners=0, losers=0, win_rate=0.0,
            total_pnl=Decimal("0"), avg_pnl=Decimal("0"),
            median_pnl=Decimal("0"), std_pnl=Decimal("0"),
            avg_holding_time=timedelta(), avg_points=Decimal("0"),
        )

    pnls = [t.net_pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total = len(trades)

    total_pnl = sum(pnls, Decimal("0"))
    avg_pnl = total_pnl / total

    # Median
    sorted_pnls = sorted(pnls)
    mid = total // 2
    if total % 2 == 0:
        median_pnl = (sorted_pnls[mid - 1] + sorted_pnls[mid]) / 2
    else:
        median_pnl = sorted_pnls[mid]

    # Standard deviation
    mean_f = float(avg_pnl)
    variance = sum((float(p) - mean_f) ** 2 for p in pnls) / total
    std_pnl = Decimal(str(round(variance ** 0.5, 2)))

    total_holding = sum((t.holding_time for t in trades), timedelta())
    avg_holding = total_holding / total

    avg_points = sum((t.points for t in trades), Decimal("0")) / total

    return TradeStats(
        count=total,
        winners=len(winners),
        losers=len(losers),
        win_rate=len(winners) / total,
        total_pnl=total_pnl,
        avg_pnl=avg_pnl,
        median_pnl=median_pnl,
        std_pnl=std_pnl,
        avg_holding_time=avg_holding,
        avg_points=avg_points,
    )


def analyze_by_side(trades: list[Trade]) -> dict[str, TradeStats]:
    """Analyze trades grouped by side (LONG/SHORT)."""
    longs = [t for t in trades if t.side == Side.BUY]
    shorts = [t for t in trades if t.side == Side.SELL]
    return {
        "long": analyze_trades(longs),
        "short": analyze_trades(shorts),
    }


def analyze_by_tag(trades: list[Trade]) -> dict[str, TradeStats]:
    """Analyze trades grouped by tag."""
    tags: dict[str, list[Trade]] = {}
    for t in trades:
        key = t.tag if t.tag else "untagged"
        tags.setdefault(key, []).append(t)
    return {tag: analyze_trades(tlist) for tag, tlist in tags.items()}


def trade_to_dict(trade: Trade) -> dict:
    """Convert a Trade to a serializable dictionary."""
    return {
        "id": str(trade.id),
        "product": trade.product.value,
        "side": trade.side.name,
        "entry_time": trade.entry_time.isoformat(),
        "entry_price": str(trade.entry_price),
        "exit_time": trade.exit_time.isoformat(),
        "exit_price": str(trade.exit_price),
        "quantity": trade.quantity,
        "points": str(trade.points),
        "pnl": str(trade.pnl),
        "commission": str(trade.commission),
        "tax": str(trade.tax),
        "net_pnl": str(trade.net_pnl),
        "holding_time": str(trade.holding_time),
        "tag": trade.tag,
    }
