"""Run the example MA Cross strategy on synthetic data."""

import sys
from pathlib import Path

# Add src and project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from decimal import Decimal

from taiex_backtest.data.feed import DataFeed
from taiex_backtest.data.synthetic import generate_ticks
from taiex_backtest.data.writer import ticks_to_dataframe, write_ticks
from taiex_backtest.domain.enums import ProductType
from taiex_backtest.engine.backtest_engine import BacktestEngine
from strategies.example_ma_cross import MACrossStrategy


def main():
    print("=" * 60)
    print("TAIEX Futures Tick Backtesting System - Example Run")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1] Generating synthetic tick data...")
    ticks = generate_ticks(
        start_price=20000.0,
        num_ticks=10000,
        seed=42,
        sigma=0.20,
    )
    print(f"    Generated {len(ticks)} ticks")
    print(f"    Price range: {min(t.price for t in ticks)} - {max(t.price for t in ticks)}")
    print(f"    Time range: {ticks[0].timestamp} - {ticks[-1].timestamp}")

    # Save to parquet
    output_path = Path(__file__).parent.parent / "data" / "processed" / "synthetic_test.parquet"
    write_ticks(ticks, output_path)
    print(f"    Saved to: {output_path}")

    # Create data feed
    df = ticks_to_dataframe(ticks)
    feed = DataFeed(df)

    # Create strategy
    strategy = MACrossStrategy(fast_period=50, slow_period=200, quantity=1)
    print(f"\n[2] Strategy: {strategy.name}")

    # Run backtest
    print("\n[3] Running backtest...")
    engine = BacktestEngine(
        feed=feed,
        strategy=strategy,
        initial_capital=Decimal("1000000"),
    )
    engine.run()

    # Print results
    trades = engine.trades
    print(f"\n[4] Results:")
    print(f"    Total ticks processed: {engine.clock.tick_count}")
    print(f"    Total trades: {len(trades)}")
    print(f"    Final capital: {engine.capital:,.0f} TWD")
    print(f"    P&L: {engine.capital - Decimal('1000000'):+,.0f} TWD")

    if trades:
        print(f"\n[5] Trade Details:")
        print(f"    {'#':>3} {'Side':>5} {'Entry':>8} {'Exit':>8} {'Points':>7} {'Net PnL':>10}")
        print(f"    {'-'*3} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")
        for i, t in enumerate(trades, 1):
            print(
                f"    {i:>3} {t.side.name:>5} {t.entry_price:>8} {t.exit_price:>8} "
                f"{t.points:>+7} {t.net_pnl:>+10,.0f}"
            )

        total_pnl = sum(t.net_pnl for t in trades)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(winners) / len(trades) * 100 if trades else 0

        print(f"\n[6] Summary Statistics:")
        print(f"    Win rate: {win_rate:.1f}%")
        print(f"    Winners: {len(winners)}, Losers: {len(losers)}")
        print(f"    Total net PnL: {total_pnl:+,.0f} TWD")
        if winners:
            print(f"    Avg winner: {sum(t.net_pnl for t in winners) / len(winners):+,.0f} TWD")
        if losers:
            print(f"    Avg loser: {sum(t.net_pnl for t in losers) / len(losers):+,.0f} TWD")

    # Generate HTML report
    if trades:
        from taiex_backtest.analytics.report import generate_report
        report_path = Path(__file__).parent.parent / "data" / "processed" / "report.html"
        generate_report(
            trades=trades,
            initial_capital=Decimal("1000000"),
            strategy_name=strategy.name,
            output_path=report_path,
        )
        print(f"\n[7] HTML report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("Backtest completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
