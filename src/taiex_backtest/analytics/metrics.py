"""Performance metrics for backtesting results."""

from dataclasses import dataclass
from decimal import Decimal
from datetime import timedelta
import math

from ..domain.models import Trade
from ..domain.enums import Side


@dataclass(frozen=True)
class PerformanceMetrics:
    """Complete performance metrics for a backtest run."""
    # Basic
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # PnL
    total_pnl: Decimal
    total_commission: Decimal
    total_tax: Decimal
    total_net_pnl: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: float

    # Averages
    avg_pnl: Decimal
    avg_winner: Decimal
    avg_loser: Decimal
    avg_holding_time: timedelta
    largest_winner: Decimal
    largest_loser: Decimal

    # Ratios
    payoff_ratio: float         # avg_winner / abs(avg_loser)
    expectancy: Decimal         # (win_rate * avg_winner) - (loss_rate * abs(avg_loser))

    # Streaks
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Risk
    max_drawdown: Decimal
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Long/Short breakdown
    long_trades: int
    short_trades: int
    long_pnl: Decimal
    short_pnl: Decimal


def calculate_metrics(
    trades: list[Trade],
    equity_curve: list[Decimal],
    initial_capital: Decimal = Decimal("1000000"),
    risk_free_rate: float = 0.02,
    annualization_factor: float = 252.0,
) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from trades and equity curve.

    Args:
        trades: List of completed Trade objects.
        equity_curve: List of equity values at each trade close.
        initial_capital: Starting capital.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.
        annualization_factor: Trading days per year.
    """
    if not trades:
        return _empty_metrics()

    # Basic counts
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    total = len(trades)
    win_rate = len(winners) / total if total > 0 else 0.0

    # PnL sums
    total_pnl = sum((t.pnl for t in trades), Decimal("0"))
    total_commission = sum((t.commission for t in trades), Decimal("0"))
    total_tax = sum((t.tax for t in trades), Decimal("0"))
    total_net_pnl = sum((t.net_pnl for t in trades), Decimal("0"))

    gross_profit = sum((t.net_pnl for t in winners), Decimal("0"))
    gross_loss = sum((t.net_pnl for t in losers), Decimal("0"))

    profit_factor = (
        float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
    )

    # Averages
    avg_pnl = total_net_pnl / total
    avg_winner = gross_profit / len(winners) if winners else Decimal("0")
    avg_loser = gross_loss / len(losers) if losers else Decimal("0")

    total_holding = sum(
        (t.holding_time for t in trades), timedelta()
    )
    avg_holding_time = total_holding / total

    largest_winner = max((t.net_pnl for t in trades), default=Decimal("0"))
    largest_loser = min((t.net_pnl for t in trades), default=Decimal("0"))

    # Payoff ratio
    payoff_ratio = (
        float(avg_winner / abs(avg_loser)) if avg_loser != 0 else float("inf")
    )

    # Expectancy
    loss_rate = 1.0 - win_rate
    expectancy = (
        Decimal(str(win_rate)) * avg_winner
        + Decimal(str(loss_rate)) * avg_loser
    )

    # Streaks
    max_wins, max_losses = _calculate_streaks(trades)

    # Drawdown from equity curve
    max_dd, max_dd_pct = _calculate_drawdown(equity_curve, initial_capital)

    # Risk-adjusted returns
    returns = _calculate_returns(equity_curve, initial_capital)
    sharpe = _sharpe_ratio(returns, risk_free_rate, annualization_factor)
    sortino = _sortino_ratio(returns, risk_free_rate, annualization_factor)

    # Calmar ratio
    total_return_pct = float(total_net_pnl / initial_capital) if initial_capital > 0 else 0.0
    calmar = total_return_pct / max_dd_pct if max_dd_pct > 0 else float("inf")

    # Long/Short
    long_trades_list = [t for t in trades if t.side == Side.BUY]
    short_trades_list = [t for t in trades if t.side == Side.SELL]

    return PerformanceMetrics(
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_commission=total_commission,
        total_tax=total_tax,
        total_net_pnl=total_net_pnl,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        avg_pnl=avg_pnl,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        avg_holding_time=avg_holding_time,
        largest_winner=largest_winner,
        largest_loser=largest_loser,
        payoff_ratio=payoff_ratio,
        expectancy=expectancy,
        max_consecutive_wins=max_wins,
        max_consecutive_losses=max_losses,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        long_trades=len(long_trades_list),
        short_trades=len(short_trades_list),
        long_pnl=sum((t.net_pnl for t in long_trades_list), Decimal("0")),
        short_pnl=sum((t.net_pnl for t in short_trades_list), Decimal("0")),
    )


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics when no trades exist."""
    return PerformanceMetrics(
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
        total_pnl=Decimal("0"), total_commission=Decimal("0"),
        total_tax=Decimal("0"), total_net_pnl=Decimal("0"),
        gross_profit=Decimal("0"), gross_loss=Decimal("0"),
        profit_factor=0.0,
        avg_pnl=Decimal("0"), avg_winner=Decimal("0"), avg_loser=Decimal("0"),
        avg_holding_time=timedelta(), largest_winner=Decimal("0"),
        largest_loser=Decimal("0"),
        payoff_ratio=0.0, expectancy=Decimal("0"),
        max_consecutive_wins=0, max_consecutive_losses=0,
        max_drawdown=Decimal("0"), max_drawdown_pct=0.0,
        sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
        long_trades=0, short_trades=0,
        long_pnl=Decimal("0"), short_pnl=Decimal("0"),
    )


def _calculate_streaks(trades: list[Trade]) -> tuple[int, int]:
    """Calculate maximum consecutive wins and losses."""
    max_wins = max_losses = 0
    current_wins = current_losses = 0

    for trade in trades:
        if trade.net_pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def _calculate_drawdown(
    equity_curve: list[Decimal],
    initial_capital: Decimal,
) -> tuple[Decimal, float]:
    """Calculate maximum drawdown in absolute and percentage terms."""
    if not equity_curve:
        return Decimal("0"), 0.0

    peak = initial_capital
    max_dd = Decimal("0")
    max_dd_pct = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = float(dd / peak) if peak > 0 else 0.0

    return max_dd, max_dd_pct


def _calculate_returns(
    equity_curve: list[Decimal],
    initial_capital: Decimal,
) -> list[float]:
    """Calculate per-trade returns from equity curve."""
    if not equity_curve:
        return []

    prev = initial_capital
    returns = []
    for equity in equity_curve:
        if prev > 0:
            returns.append(float((equity - prev) / prev))
        else:
            returns.append(0.0)
        prev = equity
    return returns


def _sharpe_ratio(
    returns: list[float],
    risk_free_rate: float,
    annualization_factor: float,
) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    rf_per_period = risk_free_rate / annualization_factor
    excess = mean_ret - rf_per_period

    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 0.0

    if std == 0:
        return 0.0

    return (excess / std) * math.sqrt(annualization_factor)


def _sortino_ratio(
    returns: list[float],
    risk_free_rate: float,
    annualization_factor: float,
) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0

    mean_ret = sum(returns) / len(returns)
    rf_per_period = risk_free_rate / annualization_factor
    excess = mean_ret - rf_per_period

    downside = [min(r, 0.0) for r in returns]
    downside_var = sum(d ** 2 for d in downside) / (len(returns) - 1)
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0

    if downside_std == 0:
        return 0.0

    return (excess / downside_std) * math.sqrt(annualization_factor)
