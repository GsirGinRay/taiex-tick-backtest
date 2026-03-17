"""Parquet schema definitions for tick data."""

import polars as pl


TICK_SCHEMA = {
    "timestamp": pl.Datetime("ns"),
    "price": pl.Float64,
    "volume": pl.Int32,
    "product": pl.Utf8,
    "session": pl.Utf8,
}

TICK_COLUMNS = list(TICK_SCHEMA.keys())


def validate_schema(df: pl.DataFrame) -> bool:
    """Validate that a DataFrame matches the tick schema."""
    for col, dtype in TICK_SCHEMA.items():
        if col not in df.columns:
            return False
        if df[col].dtype != dtype:
            return False
    return True
