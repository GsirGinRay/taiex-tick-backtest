"""FastAPI application for the backtesting system."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routers import backtest, data, ws


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TAIEX Tick Backtest",
        description="台指期 Tick 級回測系绱",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files and templates
    ui_dir = Path(__file__).parent.parent / "ui"
    static_dir = ui_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Include routers
    app.include_router(backtest.router, prefix="/api", tags=["backtest"])
    app.include_router(data.router, prefix="/api/data", tags=["data"])
    app.include_router(ws.router, tags=["websocket"])

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
