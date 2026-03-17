"""HTML report generation for backtest results."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ..domain.models import Trade
from .equity_curve import build_equity_curve, equity_curve_to_dicts, get_monthly_returns
from .metrics import PerformanceMetrics, calculate_metrics
from .trade_analyzer import analyze_trades, trade_to_dict


def generate_report(
    trades: list[Trade],
    initial_capital: Decimal = Decimal("1000000"),
    strategy_name: str = "Strategy",
    output_path: Path | str | None = None,
) -> str:
    """Generate an HTML report for backtest results.

    Args:
        trades: List of completed Trade objects.
        initial_capital: Starting capital.
        strategy_name: Name of the strategy for the report title.
        output_path: Optional path to save the HTML file.

    Returns:
        HTML string of the report.
    """
    from .equity_curve import get_equity_values

    equity_values = get_equity_values(trades, initial_capital)
    metrics = calculate_metrics(trades, equity_values, initial_capital)
    curve = build_equity_curve(trades, initial_capital)
    curve_data = equity_curve_to_dicts(curve)
    monthly = get_monthly_returns(trades, initial_capital)
    trade_stats = analyze_trades(trades)
    trade_list = [trade_to_dict(t) for t in trades]

    html = _render_html(
        strategy_name=strategy_name,
        metrics=metrics,
        curve_data=curve_data,
        monthly=monthly,
        trade_stats=trade_stats,
        trade_list=trade_list,
        initial_capital=initial_capital,
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    return html


def _render_html(
    strategy_name: str,
    metrics: PerformanceMetrics,
    curve_data: list[dict],
    monthly: dict[str, Decimal],
    trade_stats,
    trade_list: list[dict],
    initial_capital: Decimal,
) -> str:
    """Render the full HTML report."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format monthly returns for display
    monthly_rows = ""
    for month, pnl in sorted(monthly.items()):
        color = "#22c55e" if pnl >= 0 else "#ef4444"
        monthly_rows += f'<tr><td>{month}</td><td style="color:{color}">{pnl:+,.0f}</td></tr>\n'

    # Format trade list
    trade_rows = ""
    for i, t in enumerate(trade_list, 1):
        net_pnl = Decimal(t["net_pnl"])
        color = "#22c55e" if net_pnl >= 0 else "#ef4444"
        trade_rows += (
            f'<tr>'
            f'<td>{i}</td>'
            f'<td>{t["side"]}</td>'
            f'<td>{t["entry_time"][:19]}</td>'
            f'<td>{t["entry_price"]}</td>'
            f'<td>{t["exit_time"][:19]}</td>'
            f'<td>{t["exit_price"]}</td>'
            f'<td>{t["points"]}</td>'
            f'<td style="color:{color}">{net_pnl:+,.0f}</td>'
            f'<td>{t["tag"]}</td>'
            f'</tr>\n'
        )

    # Equity curve JSON for chart
    import json
    equity_json = json.dumps([
        {"x": d["timestamp"][:19], "y": float(d["equity"])}
        for d in curve_data
    ])
    drawdown_json = json.dumps([
        {"x": d["timestamp"][:19], "y": -d["drawdown_pct"] * 100}
        for d in curve_data
    ])

    return f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_name} - Backtest Report</title>
<style>
:root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #3b82f6;
         --green: #22c55e; --red: #ef4444; --border: #334155; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: var(--bg); color: var(--text); padding: 20px; }}
.container {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
.subtitle {{ color: #94a3b8; margin-bottom: 24px; font-size: 0.9rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.card {{ background: var(--card); border-radius: 8px; padding: 16px; border: 1px solid var(--border); }}
.card-label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
.card-value {{ font-size: 1.4rem; font-weight: 700; margin-top: 4px; }}
.positive {{ color: var(--green); }}
.negative {{ color: var(--red); }}
.section {{ background: var(--card); border-radius: 8px; padding: 20px; margin-bottom: 24px; border: 1px solid var(--border); }}
.section h2 {{ font-size: 1.2rem; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
th {{ color: #94a3b8; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:hover {{ background: rgba(59,130,246,0.05); }}
canvas {{ width: 100% !important; height: 300px !important; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
<h1>{strategy_name}</h1>
<p class="subtitle">Backtest Report | Generated: {generated_at} | Initial Capital: {initial_capital:,.0f} TWD</p>

<div class="grid">
  <div class="card">
    <div class="card-label">Net PnL</div>
    <div class="card-value {'positive' if metrics.total_net_pnl >= 0 else 'negative'}">{metrics.total_net_pnl:+,.0f}</div>
  </div>
  <div class="card">
    <div class="card-label">Total Trades</div>
    <div class="card-value">{metrics.total_trades}</div>
  </div>
  <div class="card">
    <div class="card-label">Win Rate</div>
    <div class="card-value">{metrics.win_rate:.1%}</div>
  </div>
  <div class="card">
    <div class="card-label">Profit Factor</div>
    <div class="card-value">{metrics.profit_factor:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Sharpe Ratio</div>
    <div class="card-value">{metrics.sharpe_ratio:.2f}</div>
  </div>
  <div class="card">
    <div class="card-label">Max Drawdown</div>
    <div class="card-value negative">{metrics.max_drawdown:,.0f} ({metrics.max_drawdown_pct:.2%})</div>
  </div>
</div>

<div class="section">
  <h2>Equity Curve</h2>
  <canvas id="equityChart"></canvas>
</div>

<div class="section">
  <h2>Drawdown</h2>
  <canvas id="ddChart"></canvas>
</div>

<div class="two-col">
  <div class="section">
    <h2>Performance Summary</h2>
    <table>
      <tr><td>Gross Profit</td><td class="positive">{metrics.gross_profit:+,.0f}</td></tr>
      <tr><td>Gross Loss</td><td class="negative">{metrics.gross_loss:+,.0f}</td></tr>
      <tr><td>Total Commission</td><td>{metrics.total_commission:,.0f}</td></tr>
      <tr><td>Total Tax</td><td>{metrics.total_tax:,.0f}</td></tr>
      <tr><td>Avg Winner</td><td class="positive">{metrics.avg_winner:+,.0f}</td></tr>
      <tr><td>Avg Loser</td><td class="negative">{metrics.avg_loser:+,.0f}</td></tr>
      <tr><td>Largest Winner</td><td class="positive">{metrics.largest_winner:+,.0f}</td></tr>
      <tr><td>Largest Loser</td><td class="negative">{metrics.largest_loser:+,.0f}</td></tr>
      <tr><td>Payoff Ratio</td><td>{metrics.payoff_ratio:.2f}</td></tr>
      <tr><td>Expectancy</td><td>{metrics.expectancy:+,.0f}</td></tr>
      <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
      <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
      <tr><td>Max Consecutive Wins</td><td>{metrics.max_consecutive_wins}</td></tr>
      <tr><td>Max Consecutive Losses</td><td>{metrics.max_consecutive_losses}</td></tr>
      <tr><td>Avg Holding Time</td><td>{metrics.avg_holding_time}</td></tr>
    </table>
  </div>
  <div class="section">
    <h2>Monthly Returns</h2>
    <table>
      <tr><th>Month</th><th>Net PnL (TWD)</th></tr>
      {monthly_rows}
    </table>
  </div>
</div>

<div class="section">
  <h2>Trade List</h2>
  <div style="overflow-x:auto;">
  <table>
    <tr><th>#</th><th>Side</th><th>Entry Time</th><th>Entry Price</th><th>Exit Time</th><th>Exit Price</th><th>Points</th><th>Net PnL</th><th>Tag</th></tr>
    {trade_rows}
  </table>
  </div>
</div>

<div class="section" style="text-align:center;color:#64748b;font-size:0.8rem;">
  Long: {metrics.long_trades} trades ({metrics.long_pnl:+,.0f} TWD) |
  Short: {metrics.short_trades} trades ({metrics.short_pnl:+,.0f} TWD)
</div>

</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
const equityData = {equity_json};
const ddData = {drawdown_json};

new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{ datasets: [{{
    label: 'Equity',
    data: equityData,
    borderColor: '#3b82f6',
    backgroundColor: 'rgba(59,130,246,0.1)',
    fill: true,
    pointRadius: 0,
    borderWidth: 2,
  }}] }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ display: true, ticks: {{ color: '#64748b', maxTicksLimit: 10 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{ datasets: [{{
    label: 'Drawdown %',
    data: ddData,
    borderColor: '#ef4444',
    backgroundColor: 'rgba(239,68,68,0.1)',
    fill: true,
    pointRadius: 0,
    borderWidth: 2,
  }}] }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ display: true, ticks: {{ color: '#64748b', maxTicksLimit: 10 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }}
    }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});
</script>
</body>
</html>"""
