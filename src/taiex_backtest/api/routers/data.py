"""Data management API endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import AppState, get_app_state
from ..schemas import DataSourceInfo

router = APIRouter()


@router.get("/sources")
async def list_data_sources(state: AppState = Depends(get_app_state)):
    """List available data sources."""
    sources = [
        DataSourceInfo(
            name="synthetic",
            type="synthetic",
            path="",
            num_ticks=0,
        )
    ]

    for f in state.get_parquet_files():
        import polars as pl
        try:
            df = pl.read_parquet(f)
            sources.append(DataSourceInfo(
                name=f.stem,
                type="parquet",
                path=str(f),
                num_ticks=len(df),
            ))
        except Exception:
            sources.append(DataSourceInfo(
                name=f.stem,
                type="parquet",
                path=str(f),
                num_ticks=0,
            ))

    return {"sources": [s.model_dump() for s in sources]}


@router.get("/sources/{name}")
async def get_data_source(name: str, state: AppState = Depends(get_app_state)):
    """Get details of a specific data source."""
    if name == "synthetic":
        return {
            "name": "synthetic",
            "type": "synthetic",
            "description": "GBM + Jump Diffusion synthetic data",
            "configurable": True,
            "params": {
                "start_price": {"type": "float", "default": 20000.0},
                "num_ticks": {"type": "int", "default": 5000, "min": 100, "max": 1000000},
                "seed": {"type": "int", "default": 42},
                "sigma": {"type": "float", "default": 0.20},
            },
        }

    for f in state.get_parquet_files():
        if f.stem == name:
            import polars as pl
            try:
                df = pl.read_parquet(f)
                return {
                    "name": f.stem,
                    "type": "parquet",
                    "path": str(f),
                    "num_ticks": len(df),
                    "columns": df.columns,
                    "time_range": {
                        "start": str(df["timestamp"][0]),
                        "end": str(df["timestamp"][-1]),
                    },
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    raise HTTPException(status_code=404, detail=f"Data source not found: {name}")
