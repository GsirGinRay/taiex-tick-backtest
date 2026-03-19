"""Phase 6: Web UI API tests."""

import pytest
from decimal import Decimal
from unittest.mock import patch

from fastapi.testclient import TestClient

from taiex_backtest.api.app import app
from taiex_backtest.api.dependencies import app_state
from taiex_backtest.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    GridSearchRequest,
    MetricsResponse,
    TradeResponse,
    StrategyListResponse,
    WebSocketMessage,
)
from taiex_backtest.strategy.base import Strategy
from taiex_backtest.domain.models import Tick


class DummyApiStrategy(Strategy):
    """Strategy for API tests (not named Test* to avoid pytest collection)."""
    def __init__(self, warmup: int = 10, hold_period: int = 5, quantity: int = 1):
        self._warmup = warmup
        self._hold_period = hold_period
        self._quantity = quantity
        self._tick_count = 0
        self._holding_for = 0
        self._in_position = False

    def on_tick(self, ctx, tick):
        self._tick_count += 1
        if self._in_position:
            self._holding_for += 1
            if self._holding_for >= self._hold_period:
                ctx.close_position(tag="exit")
                self._in_position = False
                self._holding_for = 0
        elif self._tick_count >= self._warmup and ctx.position.is_flat:
            ctx.buy(quantity=self._quantity, tag="entry")
            self._in_position = True
            self._holding_for = 0

    def on_stop(self, ctx):
        if not ctx.position.is_flat:
            ctx.close_position(tag="stop")


@pytest.fixture(autouse=True)
def register_test_strategy():
    """Register test strategy before each test, clean up after."""
    app_state.register_strategy("DummyApiStrategy", DummyApiStrategy)
    yield
    # Clean up
    if "DummyApiStrategy" in app_state.registry._strategies:
        del app_state.registry._strategies["DummyApiStrategy"]


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# ===== Health Check =====

class TestHealthCheck:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


# ===== Strategy List =====

class TestStrategyList:
    def test_list_strategies(self, client):
        response = client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        names = [s["name"] for s in data["strategies"]]
        assert "DummyApiStrategy" in names

    def test_list_strategies_format(self, client):
        response = client.get("/api/strategies")
        data = response.json()
        for s in data["strategies"]:
            assert "name" in s

# ===== Backtest Endpoint =====

class TestBacktestEndpoint:
    def test_run_backtest_basic(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "strategy_params": {"warmup": 10, "hold_period": 5},
            "num_ticks": 500,
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_name"] == "DummyApiStrategy"
        assert "metrics" in data
        assert "trades" in data
        assert "equity_curve" in data
        assert "monthly_returns" in data

    def test_backtest_metrics_format(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "num_ticks": 300,
            "seed": 42,
        })
        data = response.json()
        m = data["metrics"]
        assert "total_trades" in m
        assert "win_rate" in m
        assert "sharpe_ratio" in m
        assert "total_net_pnl" in m

    def test_backtest_strategy_not_found(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "NonExistentStrategy",
            "num_ticks": 100,
        })
        assert response.status_code == 404

    def test_backtest_invalid_product(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "product": "INVALID",
            "num_ticks": 100,
        })
        assert response.status_code == 400

    def test_backtest_with_mtx(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "product": "MTX",
            "num_ticks": 300,
            "seed": 42,
        })
        assert response.status_code == 200
        assert response.json()["product"] == "MTX"

    def test_backtest_custom_capital(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "initial_capital": 500000,
            "num_ticks": 300,
            "seed": 42,
        })
        assert response.status_code == 200
        assert response.json()["initial_capital"].startswith("500000")

    def test_backtest_trades_format(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "num_ticks": 500,
            "seed": 42,
        })
        data = response.json()
        if data["trades"]:
            t = data["trades"][0]
            assert "side" in t
            assert "entry_time" in t
            assert "entry_price" in t
            assert "exit_time" in t
            assert "net_pnl" in t
            assert "tag" in t

    def test_backtest_equity_curve_format(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "num_ticks": 500,
            "seed": 42,
        })
        data = response.json()
        if data["equity_curve"]:
            point = data["equity_curve"][0]
            assert "timestamp" in point
            assert "equity" in point

    def test_backtest_strategy_params(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "strategy_params": {"warmup": 50, "hold_period": 20},
            "num_ticks": 500,
            "seed": 42,
        })
        assert response.status_code == 200

    def test_backtest_invalid_strategy_params(self, client):
        response = client.post("/api/backtest", json={
            "strategy_name": "DummyApiStrategy",
            "strategy_params": {"invalid_param": 999},
            "num_ticks": 100,
        })
        assert response.status_code == 400

# ===== Grid Search Endpoint =====

class TestGridSearchEndpoint:
    def test_grid_search_basic(self, client):
        response = client.post("/api/grid-search", json={
            "strategy_name": "DummyApiStrategy",
            "param_grid": {"warmup": [10, 20], "hold_period": [5, 10]},
            "num_ticks": 300,
            "seed": 42,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_combinations"] == 4
        assert data["objective"] == "sharpe_ratio"
        assert len(data["results"]) == 4

    def test_grid_search_result_format(self, client):
        response = client.post("/api/grid-search", json={
            "strategy_name": "DummyApiStrategy",
            "param_grid": {"warmup": [10, 20]},
            "num_ticks": 300,
            "seed": 42,
        })
        data = response.json()
        r = data["results"][0]
        assert "rank" in r
        assert "params" in r
        assert "objective_value" in r
        assert "total_trades" in r
        assert "sharpe_ratio" in r

    def test_grid_search_strategy_not_found(self, client):
        response = client.post("/api/grid-search", json={
            "strategy_name": "NonExistent",
            "param_grid": {"warmup": [10]},
            "num_ticks": 100,
        })
        assert response.status_code == 404

    def test_grid_search_sorted(self, client):
        response = client.post("/api/grid-search", json={
            "strategy_name": "DummyApiStrategy",
            "param_grid": {"warmup": [10, 30, 50], "hold_period": [5, 10]},
            "num_ticks": 300,
            "seed": 42,
        })
        data = response.json()
        ranks = [r["rank"] for r in data["results"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_grid_search_custom_objective(self, client):
        response = client.post("/api/grid-search", json={
            "strategy_name": "DummyApiStrategy",
            "param_grid": {"warmup": [10, 20]},
            "objective": "total_net_pnl",
            "num_ticks": 300,
            "seed": 42,
        })
        assert response.status_code == 200
        assert response.json()["objective"] == "total_net_pnl"


# ===== Data Sources Endpoint =====

class TestDataSourcesEndpoint:
    def test_list_sources(self, client):
        response = client.get("/api/data/sources")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        # Should have at least synthetic
        names = [s["name"] for s in data["sources"]]
        assert "synthetic" in names

    def test_get_synthetic_source(self, client):
        response = client.get("/api/data/sources/synthetic")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "synthetic"
        assert data["type"] == "synthetic"
        assert "params" in data

    def test_get_nonexistent_source(self, client):
        response = client.get("/api/data/sources/does_not_exist")
        assert response.status_code == 404

# ===== Schema Tests =====

class TestSchemas:
    def test_backtest_request_defaults(self):
        req = BacktestRequest(strategy_name="test")
        assert req.product == "TX"
        assert req.initial_capital == 1000000.0
        assert req.data_source == "synthetic"
        assert req.num_ticks == 5000

    def test_backtest_request_validation(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BacktestRequest(strategy_name="test", initial_capital=-100)

    def test_grid_search_request_defaults(self):
        req = GridSearchRequest(
            strategy_name="test",
            param_grid={"a": [1, 2]},
        )
        assert req.objective == "sharpe_ratio"
        assert req.descending is True

    def test_websocket_message(self):
        msg = WebSocketMessage(type="progress", data={"pct": 50})
        assert msg.type == "progress"
        assert msg.data["pct"] == 50


# ===== WebSocket Tests =====

class TestWebSocket:
    def test_websocket_backtest(self, client):
        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({
                "strategy_name": "DummyApiStrategy",
                "strategy_params": {"warmup": 10, "hold_period": 5},
                "num_ticks": 300,
                "seed": 42,
            })
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] in ("complete", "error"):
                    break
            
            types = [m["type"] for m in messages]
            assert "progress" in types
            assert "complete" in types

    def test_websocket_strategy_not_found(self, client):
        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({
                "strategy_name": "NonExistent",
                "num_ticks": 100,
            })
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_websocket_complete_data(self, client):
        with client.websocket_connect("/ws/backtest") as ws:
            ws.send_json({
                "strategy_name": "DummyApiStrategy",
                "num_ticks": 300,
                "seed": 42,
            })
            complete_msg = None
            while True:
                msg = ws.receive_json()
                if msg["type"] == "complete":
                    complete_msg = msg
                    break
                if msg["type"] == "error":
                    break
            
            assert complete_msg is not None
            data = complete_msg["data"]
            assert "total_trades" in data
            assert "net_pnl" in data
            assert "sharpe_ratio" in data
            assert "equity_curve" in data


# ===== Dependencies Tests =====

class TestAppState:
    def test_register_strategy(self):
        from taiex_backtest.api.dependencies import AppState
        state = AppState()
        state.register_strategy("Test", DummyApiStrategy)
        assert "Test" in state.registry.list_strategies()

    def test_get_parquet_files_empty(self):
        from taiex_backtest.api.dependencies import AppState
        from pathlib import Path
        state = AppState()
        state.data_dir = Path("/nonexistent")
        assert state.get_parquet_files() == []


# ===== Connection Manager Tests =====

class TestConnectionManager:
    def test_connection_count_initial(self):
        from taiex_backtest.api.routers.ws import ConnectionManager
        mgr = ConnectionManager()
        assert mgr.connection_count == 0
