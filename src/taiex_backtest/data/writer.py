"""Parquet data writer using Polars."""

from pathlib import Path
from datetime import datetime
from decimal import Decimal

import polars as pl

from .schema import TICK_SCHEMA, validate_schema
from ..domain.models import Tick


def ticks_to_dataframe(ticks: list[Tick]) -> pl.DataFrame:
    """Convert a list of Tick domain objects to a Polars DataFrame."""
    if not ticks:
        return pl.DataFrame(schema=TICK_SCHEMA)

    return pl.DataFrame({
        "timestamp": [t.timestamp for t in ticks],
        "price": [float(t.price) for t in ticks],
        "volume": [t.volume for t in ticks],
        "product": [t.product.value for t in ticks],
        "session": [t.session.name for t in ticks],
    }).cast(TICK_SCHEMA)


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Write a tick DataFrame to Parquet format."""
    if not validate_schema(df):
        raise ValueError("DataFrame does not match tick schema")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    return path


def write_ticks(ticks: list[Tick], path: Path) -> Path:
    """Convert ticks to DataFrame and write to Parquet."""
    df = ticks_to_dataframe(ticks)
    return write_parquet(df, path)
