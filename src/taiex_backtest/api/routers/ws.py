"""WebSocket endpoint for real-time backtest updates."""

import asyncio
import json
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...data.feed import DataFeed
from ...data.synthetic import generate_ticks
from ...data.writer import ticks_to_dataframe
from ...domain.enums import EventType, ProductType
from ...domain.events import Event
from ...engine.backtest_engine import BacktestEngine
from ..dependencies import app_state

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def send(self, ws: WebSocket, message: dict[str, Any]) -> None:
        await ws.send_json(message)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


@router.websocket("/ws/backtest")
async def backtest_ws(ws: WebSocket):
    """WebSocket endpoint for running backtests with real-time updates.

    Client sends a JSON message with backtest config, then receives
    progress updates as the backtest runs.
    """
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()
            await _run_backtest_ws(ws, data)
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


async def _run_backtest_ws(ws: WebSocket, config: dict) -> None:
    """Run a backtest and send progress via WebSocket."""
    try:
        strategy_name = config.get("strategy_name", "")
        strategy_params = config.get("strategy_params", {})
        product_name = config.get("product", "TX")
        initial_capital = Decimal(str(config.get("initial_capital", 1000000)))
        num_ticks = config.get("num_ticks", 5000)
        seed = config.get("seed", 42)
        start_price = config.get("start_price", 20000.0)

        # Validate strategy
        try:
            strategy = app_state.registry.create(strategy_name, **strategy_params)
        except KeyError:
            await manager.send(ws, {
                "type": "error",
                "data": {"message": f"Strategy not found: {strategy_name}"},
            })
            return

        await manager.send(ws, {
            "type": "progress",
            "data": {"phase": "generating_data", "pct": 0},
        })

        # Generate data
        product_map = {"TX": ProductType.TX, "MTX": ProductType.MTX, "XMT": ProductType.XMT}
        product = product_map.get(product_name, ProductType.TX)

        ticks = generate_ticks(
            start_price=start_price,
            num_ticks=num_ticks,
            seed=seed,
        )
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)

        await manager.send(ws, {
            "type": "progress",
            "data": {"phase": "running_backtest", "pct": 10},
        })

        # Run backtest
        engine = BacktestEngine(
            feed=feed,
            strategy=strategy,
            product=product,
            initial_capital=initial_capital,
        )

        # Subscribe to events for live updates
        tick_count = 0
        total_ticks = num_ticks
        last_pct = 10

        def on_fill(event: Event):
            nonlocal tick_count
            # We don't await here since this is sync callback
            pass

        engine.event_bus.subscribe(EventType.ORDER_FILLED, on_fill)

        engine.run()

        await manager.send(ws, {
            "type": "progress",
            "data": {"phase": "calculating_metrics", "pct": 90},
        })

        # Calculate results
        from ...analytics.equity_curve import (
            build_equity_curve,
            equity_curve_to_dicts,
            get_equity_values,
            get_monthly_returns,
        )
        from ...analytics.metrics import calculate_metrics

        trades = engine.trades
        equity_values = get_equity_values(trades, initial_capital)
        metrics = calculate_metrics(trades, equity_values, initial_capital)
        curve = build_equity_curve(trades, initial_capital)
        curve_data = equity_curve_to_dicts(curve)

        # Send complete result
        await manager.send(ws, {
            "type": "complete",
            "data": {
                "strategy_name": strategy_name,
                "total_ticks": engine.clock.tick_count,
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "net_pnl": str(metrics.total_net_pnl),
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "profit_factor": metrics.profit_factor,
                "final_capital": str(engine.capital),
                "equity_curve": [
                    {"x": d["timestamp"][:19], "y": float(d["equity"])}
                    for d in curve_data
                ],
            },
        })

    except Exception as e:
        await manager.send(ws, {
            "type": "error",
            "data": {"message": str(e)},
        })
