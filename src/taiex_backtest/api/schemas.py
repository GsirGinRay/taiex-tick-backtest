"""Pydantic schemas for API request/response models."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


# ===== Request Schemas =====

class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    strategy_name: str = Field(..., description="Registered strategy name")
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    product: str = Field(default="TX", description="Product type: TX, MTX, XMT")
    initial_capital: float = Field(default=1000000.0, gt=0)
    data_source: str = Field(default="synthetic", description="synthetic or parquet file path")
    num_ticks: int = Field(default=5000, ge=100, le=1000000)
    seed: int | None = Field(default=42)
    start_price: float = Field(default=20000.0, gt=0)


class GridSearchRequest(BaseModel):
    """Request to run grid search optimization."""
    strategy_name: str
    param_grid: dict[str, list[Any]]
    product: str = "TX"
    initial_capital: float = 1000000.0
    objective: str = "sharpe_ratio"
    descending: bool = True
    data_source: str = "synthetic"
    num_ticks: int = 5000
    seed: int | None = 42
    start_price: float = 20000.0
    n_jobs: int = 1


class OptimizerRequest(BaseModel):
    """Request to run Optuna optimization."""
    strategy_name: str
    param_specs: list[dict[str, Any]]
    n_trials: int = Field(default=50, ge=1, le=10000)
    objective: str = "sharpe_ratio"
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$")
    product: str = "TX"
    initial_capital: float = 1000000.0
    data_source: str = "synthetic"
    num_ticks: int = 5000
    seed: int | None = 42
    start_price: float = 20000.0


# ===== Response Schemas =====

class TradeResponse(BaseModel):
    """Single trade in response."""
    side: str
    entry_time: str
    entry_price: str
    exit_time: str
    exit_price: str
    quantity: int
    points: str
    pnl: str
    net_pnl: str
    commission: str
    tax: str
    tag: str


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_net_pnl: str
    gross_profit: str
    gross_loss: str
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: str
    max_drawdown_pct: float
    avg_winner: str
    avg_loser: str
    largest_winner: str
    largest_loser: str
    payoff_ratio: float
    expectancy: str
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_time: str
    long_trades: int
    short_trades: int
    long_pnl: str
    short_pnl: str


class BacktestResponse(BaseModel):
    """Response from a backtest run."""
    strategy_name: str
    product: str
    initial_capital: str
    final_capital: str
    total_ticks: int
    metrics: MetricsResponse
    trades: list[TradeResponse]
    equity_curve: list[dict[str, Any]]
    monthly_returns: dict[str, str]


class GridSearchResultResponse(BaseModel):
    """Single grid search result."""
    rank: int
    params: dict[str, Any]
    objective_value: float
    total_trades: int
    win_rate: float
    net_pnl: str
    sharpe_ratio: float
    max_drawdown_pct: float


class GridSearchResponse(BaseModel):
    """Response from grid search."""
    total_combinations: int
    objective: str
    results: list[GridSearchResultResponse]


class OptimizerResultResponse(BaseModel):
    """Response from Optuna optimization."""
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    objective: str
    direction: str
    metrics: MetricsResponse
    all_trials: list[dict[str, Any]]


class StrategyInfo(BaseModel):
    """Strategy information."""
    name: str
    description: str = ""


class StrategyListResponse(BaseModel):
    """Response listing available strategies."""
    strategies: list[StrategyInfo]


class DataSourceInfo(BaseModel):
    """Data source information."""
    name: str
    type: str
    path: str = ""
    num_ticks: int = 0


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "progress", "tick", "fill", "complete", "error"
    data: dict[str, Any] = Field(default_factory=dict)
