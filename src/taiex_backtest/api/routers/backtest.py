"""Backtest API endpoints."""

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...analytics.equity_curve import (
    build_equity_curve,
    equity_curve_to_dicts,
    get_equity_values,
    get_monthly_returns,
)
from ...analytics.metrics import calculate_metrics
from ...data.feed import DataFeed
from ...data.synthetic import generate_ticks
from ...data.writer import ticks_to_dataframe
from ...domain.enums import ProductType
from ...engine.backtest_engine import BacktestEngine
from ...optimization.grid_search import grid_search, grid_search_summary
from ..dependencies import AppState, get_app_state
from ..schemas import (
    BacktestRequest,
    BacktestResponse,
    GridSearchRequest,
    GridSearchResponse,
    GridSearchResultResponse,
    MetricsResponse,
    OptimizerRequest,
    OptimizerResultResponse,
    StrategyInfo,
    StrategyListResponse,
    TradeResponse,
)

router = APIRouter()


def _get_product(name: str) -> ProductType:
    """Convert product name string to ProductType enum."""
    mapping = {"TX": ProductType.TX, "MTX": ProductType.MTX, "XMT": ProductType.XMT}
    if name not in mapping:
        raise HTTPException(status_code=400, detail=f"Unknown product: {name}")
    return mapping[name]


def _create_feed(req: BacktestRequest | GridSearchRequest) -> DataFeed:
    """Create a DataFeed from request parameters."""
    if req.data_source == "synthetic":
        ticks = generate_ticks(
            start_price=req.start_price,
            num_ticks=req.num_ticks,
            seed=req.seed,
        )
        df = ticks_to_dataframe(ticks)
        return DataFeed(df)
    else:
        from pathlib import Path
        path = Path(req.data_source)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Data file not found: {req.data_source}")
        return DataFeed(path)


def _metrics_to_response(metrics) -> MetricsResponse:
    """Convert PerformanceMetrics to response model."""
    return MetricsResponse(
        total_trades=metrics.total_trades,
        winning_trades=metrics.winning_trades,
        losing_trades=metrics.losing_trades,
        win_rate=metrics.win_rate,
        total_net_pnl=str(metrics.total_net_pnl),
        gross_profit=str(metrics.gross_profit),
        gross_loss=str(metrics.gross_loss),
        profit_factor=metrics.profit_factor,
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        calmar_ratio=metrics.calmar_ratio,
        max_drawdown=str(metrics.max_drawdown),
        max_drawdown_pct=metrics.max_drawdown_pct,
        avg_winner=str(metrics.avg_winner),
        avg_loser=str(metrics.avg_loser),
        largest_winner=str(metrics.largest_winner),
        largest_loser=str(metrics.largest_loser),
        payoff_ratio=metrics.payoff_ratio,
        expectancy=str(metrics.expectancy),
        max_consecutive_wins=metrics.max_consecutive_wins,
        max_consecutive_losses=metrics.max_consecutive_losses,
        avg_holding_time=str(metrics.avg_holding_time),
        long_trades=metrics.long_trades,
        short_trades=metrics.short_trades,
        long_pnl=str(metrics.long_pnl),
        short_pnl=str(metrics.short_pnl),
    )


def _trade_to_response(trade) -> TradeResponse:
    """Convert Trade to response model."""
    return TradeResponse(
        side=trade.side.name,
        entry_time=trade.entry_time.isoformat(),
        entry_price=str(trade.entry_price),
        exit_time=trade.exit_time.isoformat(),
        exit_price=str(trade.exit_price),
        quantity=trade.quantity,
        points=str(trade.points),
        pnl=str(trade.pnl),
        net_pnl=str(trade.net_pnl),
        commission=str(trade.commission),
        tax=str(trade.tax),
        tag=trade.tag,
    )


@router.get("/strategies", response_model=StrategyListResponse)
async def list_strategies(state: AppState = Depends(get_app_state)):
    """List all registered strategies."""
    names = state.registry.list_strategies()
    strategies = [StrategyInfo(name=n) for n in names]
    return StrategyListResponse(strategies=strategies)


@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    req: BacktestRequest,
    state: AppState = Depends(get_app_state),
):
    """Run a backtest with the specified strategy and parameters."""
    try:
        strategy = state.registry.create(req.strategy_name, **req.strategy_params)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found: {req.strategy_name}",
        )
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid strategy params: {e}")

    feed = _create_feed(req)
    product = _get_product(req.product)
    initial_capital = Decimal(str(req.initial_capital))

    engine = BacktestEngine(
        feed=feed,
        strategy=strategy,
        product=product,
        initial_capital=initial_capital,
    )
    engine.run()

    trades = engine.trades
    equity_values = get_equity_values(trades, initial_capital)
    metrics = calculate_metrics(trades, equity_values, initial_capital)
    curve = build_equity_curve(trades, initial_capital)
    curve_data = equity_curve_to_dicts(curve)
    monthly = get_monthly_returns(trades, initial_capital)

    return BacktestResponse(
        strategy_name=req.strategy_name,
        product=req.product,
        initial_capital=str(initial_capital),
        final_capital=str(engine.capital),
        total_ticks=engine.clock.tick_count,
        metrics=_metrics_to_response(metrics),
        trades=[_trade_to_response(t) for t in trades],
        equity_curve=[
            {"timestamp": d["timestamp"][:19], "equity": float(d["equity"])}
            for d in curve_data
        ],
        monthly_returns={k: str(v) for k, v in monthly.items()},
    )


@router.post("/grid-search", response_model=GridSearchResponse)
async def run_grid_search(
    req: GridSearchRequest,
    state: AppState = Depends(get_app_state),
):
    """Run grid search optimization."""
    try:
        strategy_cls = state.registry.get(req.strategy_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found: {req.strategy_name}",
        )

    def factory(**kwargs):
        return strategy_cls(**kwargs)

    feed = _create_feed(req)
    product = _get_product(req.product)
    initial_capital = Decimal(str(req.initial_capital))

    results = grid_search(
        strategy_factory=factory,
        param_grid=req.param_grid,
        feed=feed,
        product=product,
        initial_capital=initial_capital,
        objective=req.objective,
        descending=req.descending,
        n_jobs=req.n_jobs,
    )

    response_results = []
    for i, r in enumerate(results, 1):
        response_results.append(GridSearchResultResponse(
            rank=i,
            params=r.params,
            objective_value=r.objective_value,
            total_trades=r.metrics.total_trades,
            win_rate=r.metrics.win_rate,
            net_pnl=str(r.metrics.total_net_pnl),
            sharpe_ratio=r.metrics.sharpe_ratio,
            max_drawdown_pct=r.metrics.max_drawdown_pct,
        ))

    return GridSearchResponse(
        total_combinations=len(results),
        objective=req.objective,
        results=response_results,
    )


@router.post("/optimize", response_model=OptimizerResultResponse)
async def run_optimizer(
    req: OptimizerRequest,
    state: AppState = Depends(get_app_state),
):
    """Run Optuna optimization."""
    try:
        from ...optimization.optimizer import ParamSpec, optimize
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Optuna not installed. Install with: pip install taiex-backtest[optimization]",
        )

    try:
        strategy_cls = state.registry.get(req.strategy_name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy not found: {req.strategy_name}",
        )

    def factory(**kwargs):
        return strategy_cls(**kwargs)

    feed = _create_feed(req)
    product = _get_product(req.product)
    initial_capital = Decimal(str(req.initial_capital))

    param_specs = [ParamSpec(**spec) for spec in req.param_specs]

    result = optimize(
        strategy_factory=factory,
        param_specs=param_specs,
        feed=feed,
        n_trials=req.n_trials,
        objective=req.objective,
        direction=req.direction,
        product=product,
        initial_capital=initial_capital,
        seed=req.seed,
    )

    return OptimizerResultResponse(
        best_params=result.best_params,
        best_value=result.best_value,
        n_trials=result.n_trials,
        objective=req.objective,
        direction=req.direction,
        metrics=_metrics_to_response(result.best_metrics),
        all_trials=result.all_trials,
    )
