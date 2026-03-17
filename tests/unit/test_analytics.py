"""Tests for analytics layer: metrics, trade analyzer, equity curve, comparison, report."""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from taiex_backtest.domain.enums import ProductType, Side
from taiex_backtest.domain.models import Trade
from taiex_backtest.analytics.metrics import (
    calculate_metrics,
    _calculate_streaks,
    _calculate_drawdown,
)
from taiex_backtest.analytics.trade_analyzer import (
    analyze_trades,
    analyze_by_side,
    analyze_by_tag,
    trade_to_dict,
)
from taiex_backtest.analytics.equity_curve import (
    build_equity_curve,
    get_equity_values,
    get_monthly_returns,
    equity_curve_to_dicts,
)
from taiex_backtest.analytics.comparison import (
    compare_strategies,
    rank_by,
    comparison_to_table,
)
from taiex_backtest.analytics.report import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    side=Side.BUY,
    entry_price="20000",
    exit_price="20050",
    quantity=1,
    pnl="10000",
    commission="120",
    tax="16",
    entry_time=None,
    exit_time=None,
    tag="",
) -> Trade:
    """Helper to create test trades with sensible defaults."""
    if entry_time is None:
        entry_time = datetime(2024, 1, 2, 9, 0, 0)
    if exit_time is None:
        exit_time = datetime(2024, 1, 2, 10, 0, 0)
    return Trade(
        id=uuid4(),
        product=ProductType.TX,
        side=side,
        entry_time=entry_time,
        entry_price=Decimal(entry_price),
        exit_time=exit_time,
        exit_price=Decimal(exit_price),
        quantity=quantity,
        pnl=Decimal(pnl),
        commission=Decimal(commission),
        tax=Decimal(tax),
        tag=tag,
    )


def _sample_trades() -> list[Trade]:
    """Create a reproducible sample set of trades for testing.

    Trade 1: BUY  +10000 pnl, net = 10000-120-16 = +9864, tag=entry1
    Trade 2: BUY  -6000 pnl,  net = -6000-120-16 = -6136, tag=entry2
    Trade 3: BUY  +20000 pnl, net = 20000-120-16 = +19864, tag=entry1
    Trade 4: SELL +8000 pnl,  net = 8000-120-16 = +7864, tag=short1
    """
    return [
        _make_trade(
            pnl="10000", commission="120", tax="16",
            entry_time=datetime(2024, 1, 2, 9, 0),
            exit_time=datetime(2024, 1, 2, 10, 0),
            tag="entry1",
        ),
        _make_trade(
            pnl="-6000", commission="120", tax="16",
            entry_price="20050", exit_price="20020",
            entry_time=datetime(2024, 1, 2, 11, 0),
            exit_time=datetime(2024, 1, 2, 12, 0),
            tag="entry2",
        ),
        _make_trade(
            pnl="20000", commission="120", tax="16",
            entry_price="20020", exit_price="20120",
            entry_time=datetime(2024, 1, 3, 9, 0),
            exit_time=datetime(2024, 1, 3, 11, 0),
            tag="entry1",
        ),
        _make_trade(
            side=Side.SELL,
            pnl="8000", commission="120", tax="16",
            entry_price="20120", exit_price="20080",
            entry_time=datetime(2024, 1, 3, 13, 0),
            exit_time=datetime(2024, 1, 3, 13, 30),
            tag="short1",
        ),
    ]


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for calculate_metrics()."""

    def test_empty_trades_returns_zeroed_metrics(self):
        m = calculate_metrics([], [])
        assert m.total_trades == 0
        assert m.winning_trades == 0
        assert m.losing_trades == 0
        assert m.win_rate == 0.0
        assert m.total_net_pnl == Decimal("0")
        assert m.profit_factor == 0.0
        assert m.sharpe_ratio == 0.0
        assert m.sortino_ratio == 0.0
        assert m.calmar_ratio == 0.0
        assert m.max_drawdown == Decimal("0")
        assert m.max_drawdown_pct == 0.0
        assert m.max_consecutive_wins == 0
        assert m.max_consecutive_losses == 0
        assert m.avg_holding_time == timedelta()

    def test_basic_counts(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.total_trades == 4
        # Trade 1: net +9864 > 0 (win), Trade 2: net -6136 <= 0 (loss),
        # Trade 3: net +19864 > 0 (win), Trade 4: net +7864 > 0 (win)
        assert m.winning_trades == 3
        assert m.losing_trades == 1

    def test_win_rate(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.win_rate == pytest.approx(0.75, abs=1e-6)

    def test_pnl_sums(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)

        expected_total_pnl = Decimal("10000") + Decimal("-6000") + Decimal("20000") + Decimal("8000")
        assert m.total_pnl == expected_total_pnl  # 32000
        assert m.total_commission == Decimal("480")  # 4 * 120
        assert m.total_tax == Decimal("64")  # 4 * 16

        expected_net = sum(t.net_pnl for t in trades)
        assert m.total_net_pnl == expected_net

    def test_gross_profit_and_loss(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        # Gross profit = sum of net_pnl for winners (net > 0)
        # Winners: 9864 + 19864 + 7864 = 37592
        assert m.gross_profit == Decimal("9864") + Decimal("19864") + Decimal("7864")
        # Gross loss = sum of net_pnl for losers (net <= 0)
        # Losers: -6136
        assert m.gross_loss == Decimal("-6136")

    def test_profit_factor_positive(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        # profit_factor = gross_profit / abs(gross_loss)
        expected = float(Decimal("37592") / Decimal("6136"))
        assert m.profit_factor == pytest.approx(expected, rel=1e-4)
        assert m.profit_factor > 1.0

    def test_profit_factor_no_losers(self):
        trades = [_make_trade(pnl="5000"), _make_trade(pnl="3000")]
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.profit_factor == float("inf")

    def test_averages(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)

        expected_avg = sum(t.net_pnl for t in trades) / 4
        assert m.avg_pnl == expected_avg

        # avg_winner = gross_profit / 3
        expected_avg_winner = (Decimal("9864") + Decimal("19864") + Decimal("7864")) / 3
        assert m.avg_winner == expected_avg_winner

        # avg_loser = gross_loss / 1
        assert m.avg_loser == Decimal("-6136")

    def test_avg_holding_time(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        # Trade 1: 1h, Trade 2: 1h, Trade 3: 2h, Trade 4: 30min
        total = timedelta(hours=1) + timedelta(hours=1) + timedelta(hours=2) + timedelta(minutes=30)
        expected_avg = total / 4
        assert m.avg_holding_time == expected_avg

    def test_largest_winner_and_loser(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.largest_winner == Decimal("19864")
        assert m.largest_loser == Decimal("-6136")

    def test_payoff_ratio(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        expected = float(m.avg_winner / abs(m.avg_loser))
        assert m.payoff_ratio == pytest.approx(expected, rel=1e-4)

    def test_expectancy(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        # expectancy = win_rate * avg_winner + loss_rate * avg_loser
        expected = Decimal("0.75") * m.avg_winner + Decimal("0.25") * m.avg_loser
        assert m.expectancy == expected

    def test_long_short_breakdown(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.long_trades == 3
        assert m.short_trades == 1
        # long_pnl = net pnl of BUY trades
        long_pnl = sum(t.net_pnl for t in trades if t.side == Side.BUY)
        short_pnl = sum(t.net_pnl for t in trades if t.side == Side.SELL)
        assert m.long_pnl == long_pnl
        assert m.short_pnl == short_pnl

    def test_sharpe_ratio_is_float(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert isinstance(m.sharpe_ratio, float)

    def test_sortino_ratio_is_float(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert isinstance(m.sortino_ratio, float)

    def test_calmar_ratio_is_float(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert isinstance(m.calmar_ratio, float)

    def test_drawdown_nonnegative(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.max_drawdown >= Decimal("0")
        assert m.max_drawdown_pct >= 0.0

    def test_streaks(self):
        trades = _sample_trades()
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        # Sequence: win, loss, win, win -> max_wins=2, max_losses=1
        assert m.max_consecutive_wins == 2
        assert m.max_consecutive_losses == 1

    def test_frozen_dataclass(self):
        m = calculate_metrics([], [])
        with pytest.raises(AttributeError):
            m.total_trades = 10

    def test_single_winning_trade(self):
        trades = [_make_trade(pnl="5000")]
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.total_trades == 1
        assert m.winning_trades == 1
        assert m.losing_trades == 0
        assert m.win_rate == 1.0
        assert m.max_consecutive_wins == 1
        assert m.max_consecutive_losses == 0

    def test_single_losing_trade(self):
        trades = [_make_trade(pnl="-5000")]
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.total_trades == 1
        assert m.winning_trades == 0
        assert m.losing_trades == 1
        assert m.win_rate == 0.0
        assert m.max_consecutive_wins == 0
        assert m.max_consecutive_losses == 1

    def test_zero_pnl_trade_counted_as_loser(self):
        """A trade with exactly zero net PnL is treated as a loser (net_pnl <= 0)."""
        trade = _make_trade(pnl="136", commission="120", tax="16")  # net = 0
        equity = get_equity_values([trade])
        m = calculate_metrics([trade], equity)
        assert m.winning_trades == 0
        assert m.losing_trades == 1

    def test_custom_initial_capital(self):
        trades = _sample_trades()
        cap = Decimal("2000000")
        equity = get_equity_values(trades, initial_capital=cap)
        m = calculate_metrics(trades, equity, initial_capital=cap)
        # Same trade counts regardless of capital
        assert m.total_trades == 4
        # Drawdown pct should differ since capital is different
        assert isinstance(m.max_drawdown_pct, float)

    def test_all_losers(self):
        trades = [
            _make_trade(pnl="-2000"),
            _make_trade(pnl="-3000"),
        ]
        equity = get_equity_values(trades)
        m = calculate_metrics(trades, equity)
        assert m.winning_trades == 0
        assert m.losing_trades == 2
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0  # gross_profit=0, so 0/abs(loss)=0... actually inf guard
        assert m.gross_profit == Decimal("0")
        assert m.avg_winner == Decimal("0")


# ---------------------------------------------------------------------------
# TestCalculateStreaks
# ---------------------------------------------------------------------------

class TestCalculateStreaks:
    """Tests for _calculate_streaks() helper."""

    def test_empty_trades(self):
        wins, losses = _calculate_streaks([])
        assert wins == 0
        assert losses == 0

    def test_all_winners(self):
        trades = [_make_trade(pnl="1000") for _ in range(5)]
        wins, losses = _calculate_streaks(trades)
        assert wins == 5
        assert losses == 0

    def test_all_losers(self):
        trades = [_make_trade(pnl="-1000") for _ in range(3)]
        wins, losses = _calculate_streaks(trades)
        assert wins == 0
        assert losses == 3

    def test_alternating_win_loss(self):
        trades = [
            _make_trade(pnl="1000"),
            _make_trade(pnl="-1000"),
            _make_trade(pnl="1000"),
        ]
        wins, losses = _calculate_streaks(trades)
        assert wins == 1
        assert losses == 1

    def test_streak_at_end(self):
        trades = [
            _make_trade(pnl="-500"),
            _make_trade(pnl="1000"),
            _make_trade(pnl="2000"),
            _make_trade(pnl="3000"),
        ]
        wins, losses = _calculate_streaks(trades)
        assert wins == 3
        assert losses == 1

    def test_streak_at_beginning(self):
        trades = [
            _make_trade(pnl="1000"),
            _make_trade(pnl="2000"),
            _make_trade(pnl="-500"),
            _make_trade(pnl="-300"),
        ]
        wins, losses = _calculate_streaks(trades)
        assert wins == 2
        assert losses == 2

    def test_single_trade_winner(self):
        trades = [_make_trade(pnl="1000")]
        wins, losses = _calculate_streaks(trades)
        assert wins == 1
        assert losses == 0

    def test_single_trade_loser(self):
        trades = [_make_trade(pnl="-1000")]
        wins, losses = _calculate_streaks(trades)
        assert wins == 0
        assert losses == 1

    def test_zero_pnl_counted_as_loss_streak(self):
        """A trade with net_pnl == 0 or < 0 increments loss streak."""
        trades = [
            _make_trade(pnl="136", commission="120", tax="16"),  # net=0, loss
            _make_trade(pnl="-500"),  # loss
        ]
        wins, losses = _calculate_streaks(trades)
        assert losses == 2


# ---------------------------------------------------------------------------
# TestCalculateDrawdown
# ---------------------------------------------------------------------------

class TestCalculateDrawdown:
    """Tests for _calculate_drawdown() helper."""

    def test_empty_equity(self):
        dd, dd_pct = _calculate_drawdown([], Decimal("1000000"))
        assert dd == Decimal("0")
        assert dd_pct == 0.0

    def test_monotonically_increasing(self):
        equity = [Decimal("1010000"), Decimal("1020000"), Decimal("1030000")]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("0")
        assert dd_pct == 0.0

    def test_simple_drawdown(self):
        equity = [
            Decimal("1010000"),  # new peak 1010000
            Decimal("1000000"),  # dd = 10000 from 1010000
            Decimal("1020000"),  # new peak
        ]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("10000")
        expected_pct = float(Decimal("10000") / Decimal("1010000"))
        assert dd_pct == pytest.approx(expected_pct, abs=1e-8)

    def test_drawdown_from_initial_capital_peak(self):
        """If equity drops below initial capital, the peak is still initial_capital."""
        equity = [Decimal("990000")]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("10000")
        expected_pct = float(Decimal("10000") / Decimal("1000000"))
        assert dd_pct == pytest.approx(expected_pct, abs=1e-8)

    def test_multiple_drawdowns_returns_max(self):
        equity = [
            Decimal("1010000"),   # peak 1010000
            Decimal("1005000"),   # dd = 5000
            Decimal("1020000"),   # peak 1020000
            Decimal("1000000"),   # dd = 20000 (max)
            Decimal("1030000"),   # peak 1030000
        ]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("20000")

    def test_single_point_no_drawdown(self):
        equity = [Decimal("1050000")]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("0")

    def test_continuous_decline(self):
        equity = [Decimal("990000"), Decimal("980000"), Decimal("970000")]
        dd, dd_pct = _calculate_drawdown(equity, Decimal("1000000"))
        assert dd == Decimal("30000")


# ---------------------------------------------------------------------------
# TestTradeAnalyzer
# ---------------------------------------------------------------------------

class TestTradeAnalyzer:
    """Tests for analyze_trades() and related functions."""

    def test_analyze_empty(self):
        stats = analyze_trades([])
        assert stats.count == 0
        assert stats.winners == 0
        assert stats.losers == 0
        assert stats.win_rate == 0.0
        assert stats.total_pnl == Decimal("0")
        assert stats.avg_pnl == Decimal("0")
        assert stats.median_pnl == Decimal("0")
        assert stats.std_pnl == Decimal("0")
        assert stats.avg_holding_time == timedelta()
        assert stats.avg_points == Decimal("0")

    def test_analyze_basic_counts(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        assert stats.count == 4
        assert stats.winners == 3
        assert stats.losers == 1

    def test_analyze_win_rate(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        assert stats.win_rate == pytest.approx(0.75, abs=1e-6)

    def test_analyze_total_pnl(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        expected = sum(t.net_pnl for t in trades)
        assert stats.total_pnl == expected

    def test_analyze_avg_pnl(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        expected = sum(t.net_pnl for t in trades) / 4
        assert stats.avg_pnl == expected

    def test_median_pnl_odd_count(self):
        trades = _sample_trades()[:3]  # 3 trades
        stats = analyze_trades(trades)
        assert isinstance(stats.median_pnl, Decimal)
        # net pnls sorted: -6136, 9864, 19864 -> median = 9864
        assert stats.median_pnl == Decimal("9864")

    def test_median_pnl_even_count(self):
        trades = _sample_trades()  # 4 trades
        stats = analyze_trades(trades)
        assert isinstance(stats.median_pnl, Decimal)
        # net pnls sorted: -6136, 7864, 9864, 19864 -> median = (7864+9864)/2 = 8864
        assert stats.median_pnl == Decimal("8864")

    def test_std_pnl_type(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        assert isinstance(stats.std_pnl, Decimal)
        assert stats.std_pnl >= Decimal("0")

    def test_avg_holding_time(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        total = sum((t.holding_time for t in trades), timedelta())
        expected = total / len(trades)
        assert stats.avg_holding_time == expected

    def test_avg_points(self):
        trades = _sample_trades()
        stats = analyze_trades(trades)
        total_points = sum(t.points for t in trades)
        expected = total_points / len(trades)
        assert stats.avg_points == expected

    def test_single_trade(self):
        trade = _make_trade(pnl="5000")
        stats = analyze_trades([trade])
        assert stats.count == 1
        assert stats.median_pnl == trade.net_pnl
        assert stats.std_pnl == Decimal("0")  # single value -> 0 variance

    def test_analyze_frozen_dataclass(self):
        stats = analyze_trades([])
        with pytest.raises(AttributeError):
            stats.count = 99


class TestAnalyzeBySide:
    """Tests for analyze_by_side()."""

    def test_returns_long_and_short_keys(self):
        trades = _sample_trades()
        result = analyze_by_side(trades)
        assert "long" in result
        assert "short" in result

    def test_long_short_counts(self):
        trades = _sample_trades()
        result = analyze_by_side(trades)
        assert result["long"].count == 3
        assert result["short"].count == 1

    def test_empty_trades(self):
        result = analyze_by_side([])
        assert result["long"].count == 0
        assert result["short"].count == 0

    def test_all_long(self):
        trades = [_make_trade(side=Side.BUY) for _ in range(3)]
        result = analyze_by_side(trades)
        assert result["long"].count == 3
        assert result["short"].count == 0

    def test_all_short(self):
        trades = [_make_trade(side=Side.SELL) for _ in range(2)]
        result = analyze_by_side(trades)
        assert result["long"].count == 0
        assert result["short"].count == 2


class TestAnalyzeByTag:
    """Tests for analyze_by_tag()."""

    def test_groups_by_tag(self):
        trades = _sample_trades()
        result = analyze_by_tag(trades)
        assert "entry1" in result
        assert "entry2" in result
        assert "short1" in result
        assert result["entry1"].count == 2
        assert result["entry2"].count == 1
        assert result["short1"].count == 1

    def test_empty_tag_grouped_as_untagged(self):
        trades = [_make_trade(tag=""), _make_trade(tag="")]
        result = analyze_by_tag(trades)
        assert "untagged" in result
        assert result["untagged"].count == 2

    def test_empty_trades(self):
        result = analyze_by_tag([])
        assert result == {}

    def test_mixed_tagged_and_untagged(self):
        trades = [
            _make_trade(tag="alpha"),
            _make_trade(tag=""),
            _make_trade(tag="alpha"),
        ]
        result = analyze_by_tag(trades)
        assert result["alpha"].count == 2
        assert result["untagged"].count == 1


class TestTradeToDict:
    """Tests for trade_to_dict()."""

    def test_keys_present(self):
        trade = _sample_trades()[0]
        d = trade_to_dict(trade)
        expected_keys = {
            "id", "product", "side", "entry_time", "entry_price",
            "exit_time", "exit_price", "quantity", "points", "pnl",
            "commission", "tax", "net_pnl", "holding_time", "tag",
        }
        assert set(d.keys()) == expected_keys

    def test_side_is_name(self):
        trade = _make_trade(side=Side.BUY)
        d = trade_to_dict(trade)
        assert d["side"] == "BUY"

        trade_sell = _make_trade(side=Side.SELL)
        d_sell = trade_to_dict(trade_sell)
        assert d_sell["side"] == "SELL"

    def test_product_is_value(self):
        trade = _make_trade()
        d = trade_to_dict(trade)
        assert d["product"] == "TX"

    def test_prices_are_strings(self):
        trade = _make_trade(entry_price="20000", exit_price="20050")
        d = trade_to_dict(trade)
        assert d["entry_price"] == "20000"
        assert d["exit_price"] == "20050"

    def test_net_pnl_in_dict(self):
        trade = _make_trade(pnl="10000", commission="120", tax="16")
        d = trade_to_dict(trade)
        assert d["net_pnl"] == str(trade.net_pnl)

    def test_id_is_string(self):
        trade = _make_trade()
        d = trade_to_dict(trade)
        assert isinstance(d["id"], str)

    def test_times_are_isoformat(self):
        entry = datetime(2024, 1, 2, 9, 30, 0)
        exit_ = datetime(2024, 1, 2, 10, 45, 0)
        trade = _make_trade(entry_time=entry, exit_time=exit_)
        d = trade_to_dict(trade)
        assert d["entry_time"] == entry.isoformat()
        assert d["exit_time"] == exit_.isoformat()

    def test_tag_preserved(self):
        trade = _make_trade(tag="my_signal")
        d = trade_to_dict(trade)
        assert d["tag"] == "my_signal"


# ---------------------------------------------------------------------------
# TestEquityCurve
# ---------------------------------------------------------------------------

class TestEquityCurve:
    """Tests for build_equity_curve() and related functions."""

    def test_empty_trades(self):
        points = build_equity_curve([])
        assert points == []

    def test_curve_length_matches_trades(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        assert len(points) == 4

    def test_first_point_equity(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        expected_equity = Decimal("1000000") + trades[0].net_pnl
        assert points[0].equity == expected_equity

    def test_cumulative_equity(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        cumulative = Decimal("1000000")
        for i, trade in enumerate(trades):
            cumulative += trade.net_pnl
            assert points[i].equity == cumulative

    def test_timestamp_is_exit_time(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        for i, trade in enumerate(trades):
            assert points[i].timestamp == trade.exit_time

    def test_trade_pnl_field(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        for i, trade in enumerate(trades):
            assert points[i].trade_pnl == trade.net_pnl

    def test_drawdown_tracking_after_loss(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        # After trade 2 (loss), there should be a positive drawdown
        # Trade 1 pushes equity to 1009864 (peak)
        # Trade 2 pushes equity to 1009864 - 6136 = 1003728, dd = 6136
        assert points[1].drawdown == Decimal("6136")
        assert points[1].drawdown_pct > 0.0

    def test_no_drawdown_when_always_rising(self):
        trades = [
            _make_trade(pnl="5000"),
            _make_trade(pnl="3000"),
        ]
        points = build_equity_curve(trades)
        assert points[0].drawdown == Decimal("0")
        assert points[1].drawdown == Decimal("0")

    def test_custom_initial_capital(self):
        trades = [_make_trade(pnl="1000")]
        cap = Decimal("500000")
        points = build_equity_curve(trades, initial_capital=cap)
        assert points[0].equity == cap + trades[0].net_pnl

    def test_equity_point_frozen(self):
        trades = [_make_trade(pnl="1000")]
        points = build_equity_curve(trades)
        with pytest.raises(AttributeError):
            points[0].equity = Decimal("9999999")


class TestGetEquityValues:
    """Tests for get_equity_values()."""

    def test_empty_trades(self):
        values = get_equity_values([])
        assert values == []

    def test_length_matches_trades(self):
        trades = _sample_trades()
        values = get_equity_values(trades)
        assert len(values) == 4

    def test_all_decimal(self):
        trades = _sample_trades()
        values = get_equity_values(trades)
        assert all(isinstance(v, Decimal) for v in values)

    def test_values_match_build_equity_curve(self):
        trades = _sample_trades()
        values = get_equity_values(trades)
        points = build_equity_curve(trades)
        for v, p in zip(values, points):
            assert v == p.equity

    def test_custom_initial_capital(self):
        trades = [_make_trade(pnl="5000")]
        cap = Decimal("2000000")
        values = get_equity_values(trades, initial_capital=cap)
        expected = cap + trades[0].net_pnl
        assert values[0] == expected


class TestGetMonthlyReturns:
    """Tests for get_monthly_returns()."""

    def test_empty_trades(self):
        monthly = get_monthly_returns([])
        assert monthly == {}

    def test_single_month(self):
        trades = _sample_trades()  # all in 2024-01
        monthly = get_monthly_returns(trades)
        assert "2024-01" in monthly
        total_net = sum(t.net_pnl for t in trades)
        assert monthly["2024-01"] == total_net

    def test_multiple_months(self):
        trades = [
            _make_trade(pnl="5000", exit_time=datetime(2024, 1, 15, 10, 0)),
            _make_trade(pnl="3000", exit_time=datetime(2024, 2, 10, 10, 0)),
            _make_trade(pnl="-2000", exit_time=datetime(2024, 2, 20, 10, 0)),
        ]
        monthly = get_monthly_returns(trades)
        assert "2024-01" in monthly
        assert "2024-02" in monthly
        assert monthly["2024-01"] == trades[0].net_pnl
        assert monthly["2024-02"] == trades[1].net_pnl + trades[2].net_pnl

    def test_total_monthly_equals_total_pnl(self):
        trades = _sample_trades()
        monthly = get_monthly_returns(trades)
        total_monthly = sum(monthly.values())
        total_net = sum(t.net_pnl for t in trades)
        assert total_monthly == total_net


class TestEquityCurveToDicts:
    """Tests for equity_curve_to_dicts()."""

    def test_empty_curve(self):
        dicts = equity_curve_to_dicts([])
        assert dicts == []

    def test_length_matches(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        dicts = equity_curve_to_dicts(points)
        assert len(dicts) == 4

    def test_keys_present(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        dicts = equity_curve_to_dicts(points)
        expected_keys = {"timestamp", "equity", "drawdown", "drawdown_pct", "trade_pnl"}
        for d in dicts:
            assert set(d.keys()) == expected_keys

    def test_timestamp_is_isoformat_string(self):
        trades = [_make_trade(exit_time=datetime(2024, 3, 1, 14, 30))]
        points = build_equity_curve(trades)
        dicts = equity_curve_to_dicts(points)
        assert dicts[0]["timestamp"] == datetime(2024, 3, 1, 14, 30).isoformat()

    def test_equity_is_string(self):
        trades = [_make_trade(pnl="5000")]
        points = build_equity_curve(trades)
        dicts = equity_curve_to_dicts(points)
        assert isinstance(dicts[0]["equity"], str)

    def test_drawdown_pct_is_rounded_float(self):
        trades = _sample_trades()
        points = build_equity_curve(trades)
        dicts = equity_curve_to_dicts(points)
        for d in dicts:
            assert isinstance(d["drawdown_pct"], float)


# ---------------------------------------------------------------------------
# TestComparison
# ---------------------------------------------------------------------------

class TestComparison:
    """Tests for compare_strategies(), rank_by(), comparison_to_table()."""

    @staticmethod
    def _make_metrics_from_trades(pnl: str):
        trades = [_make_trade(pnl=pnl)]
        equity = get_equity_values(trades)
        return calculate_metrics(trades, equity)

    def test_compare_strategies_count(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("-5000")
        comp = compare_strategies({"A": m1, "B": m2})
        assert comp.count == 2

    def test_compare_strategies_names(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("-5000")
        comp = compare_strategies({"A": m1, "B": m2})
        assert comp.names == ["A", "B"]

    def test_compare_strategies_metrics(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("-5000")
        comp = compare_strategies({"A": m1, "B": m2})
        assert len(comp.metrics) == 2
        assert comp.metrics[0] is m1
        assert comp.metrics[1] is m2

    def test_single_strategy(self):
        m = self._make_metrics_from_trades("10000")
        comp = compare_strategies({"Only": m})
        assert comp.count == 1
        assert comp.names == ["Only"]

    def test_empty_strategies(self):
        comp = compare_strategies({})
        assert comp.count == 0
        assert comp.names == []

    def test_rank_by_total_net_pnl_descending(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("20000")
        m3 = self._make_metrics_from_trades("5000")
        comp = compare_strategies({"A": m1, "B": m2, "C": m3})
        ranked = rank_by(comp, "total_net_pnl")
        # B has highest net pnl, then A, then C
        assert ranked[0][0] == "B"
        assert ranked[-1][0] == "C"

    def test_rank_by_ascending(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("20000")
        comp = compare_strategies({"A": m1, "B": m2})
        ranked = rank_by(comp, "total_net_pnl", descending=False)
        assert ranked[0][0] == "A"

    def test_rank_by_win_rate(self):
        m1 = self._make_metrics_from_trades("10000")  # 1 winner -> 100%
        m2 = self._make_metrics_from_trades("-5000")   # 0 winners -> 0%
        comp = compare_strategies({"A": m1, "B": m2})
        ranked = rank_by(comp, "win_rate")
        assert ranked[0][0] == "A"

    def test_rank_by_total_trades(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("-5000")
        comp = compare_strategies({"A": m1, "B": m2})
        ranked = rank_by(comp, "total_trades")
        # Both have 1 trade, so order doesn't matter, just check structure
        assert len(ranked) == 2
        assert isinstance(ranked[0], tuple)
        assert len(ranked[0]) == 2

    def test_comparison_to_table_structure(self):
        m1 = self._make_metrics_from_trades("10000")
        comp = compare_strategies({"StratA": m1})
        table = comparison_to_table(comp)
        assert len(table) > 0
        assert "metric" in table[0]
        assert "StratA" in table[0]

    def test_comparison_to_table_all_key_metrics(self):
        m1 = self._make_metrics_from_trades("10000")
        comp = compare_strategies({"A": m1})
        table = comparison_to_table(comp)
        metric_labels = [row["metric"] for row in table]
        expected_labels = [
            "Total Trades", "Win Rate", "Net PnL", "Profit Factor",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Max DD %",
            "Calmar Ratio", "Expectancy", "Payoff Ratio",
            "Max Consecutive Wins", "Max Consecutive Losses",
        ]
        assert metric_labels == expected_labels

    def test_comparison_to_table_multiple_strategies(self):
        m1 = self._make_metrics_from_trades("10000")
        m2 = self._make_metrics_from_trades("-5000")
        comp = compare_strategies({"Alpha": m1, "Beta": m2})
        table = comparison_to_table(comp)
        for row in table:
            assert "Alpha" in row
            assert "Beta" in row

    def test_comparison_to_table_decimal_as_string(self):
        m1 = self._make_metrics_from_trades("10000")
        comp = compare_strategies({"A": m1})
        table = comparison_to_table(comp)
        # "Net PnL" (total_net_pnl) is Decimal -> should be str
        pnl_row = next(r for r in table if r["metric"] == "Net PnL")
        assert isinstance(pnl_row["A"], str)

    def test_comparison_to_table_float_formatted(self):
        m1 = self._make_metrics_from_trades("10000")
        comp = compare_strategies({"A": m1})
        table = comparison_to_table(comp)
        # "Win Rate" is float -> should be formatted "X.XXXX"
        wr_row = next(r for r in table if r["metric"] == "Win Rate")
        assert "." in wr_row["A"]


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------

class TestReport:
    """Tests for generate_report()."""

    def test_generate_report_returns_html_string(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Test Strategy")
        assert isinstance(html, str)
        assert "<html" in html

    def test_strategy_name_in_report(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="My Alpha Strategy")
        assert "My Alpha Strategy" in html

    def test_empty_trades_produces_valid_html(self):
        html = generate_report([], strategy_name="Empty")
        assert "<html" in html
        assert "Empty" in html

    def test_report_contains_equity_section(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Test")
        assert "Equity" in html

    def test_report_contains_key_metrics(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Data Check")
        assert "Win Rate" in html
        assert "Sharpe" in html
        assert "Drawdown" in html
        assert "Profit Factor" in html

    def test_report_contains_trade_list(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Trades Check")
        assert "Trade List" in html
        assert "BUY" in html
        assert "SELL" in html

    def test_report_contains_monthly_returns(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Monthly Check")
        assert "Monthly" in html
        assert "2024-01" in html

    def test_report_to_file(self, tmp_path: Path):
        trades = _sample_trades()
        output = tmp_path / "report.html"
        html = generate_report(trades, output_path=output, strategy_name="File Test")
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert content == html

    def test_report_to_file_creates_parent_dirs(self, tmp_path: Path):
        trades = _sample_trades()
        output = tmp_path / "subdir" / "nested" / "report.html"
        html = generate_report(trades, output_path=output, strategy_name="Nested")
        assert output.exists()
        assert "<html" in output.read_text(encoding="utf-8")

    def test_report_custom_initial_capital(self):
        trades = _sample_trades()
        cap = Decimal("2000000")
        html = generate_report(trades, initial_capital=cap, strategy_name="Cap Test")
        assert "2,000,000" in html

    def test_report_long_short_summary(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Side Check")
        assert "Long" in html
        assert "Short" in html

    def test_report_chart_data_embedded(self):
        trades = _sample_trades()
        html = generate_report(trades, strategy_name="Chart Check")
        # Chart.js is included
        assert "chart.js" in html.lower() or "Chart" in html
        # Equity and drawdown data should be embedded as JSON arrays
        assert "equityData" in html
        assert "ddData" in html

    def test_report_returns_same_html_as_file(self, tmp_path: Path):
        """The returned HTML string matches what is written to the file."""
        trades = _sample_trades()
        output = tmp_path / "test.html"
        html = generate_report(trades, output_path=output, strategy_name="Match Test")
        file_content = output.read_text(encoding="utf-8")
        assert html == file_content
