"""Data feed for streaming tick data to the backtesting engine."""

from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl

from ..domain.enums import ProductType, Session
from ..domain.models import Tick


class DataFeed:
    """Streams tick data from a Parquet file or DataFrame."""

    def __init__(
        self,
        source: pl.DataFrame | Path | str,
        product: ProductType = ProductType.TX,
    ):
        if isinstance(source, (str, Path)):
            self._df = pl.read_parquet(source)
        else:
            self._df = source

        self._product = product
        self._df = self._df.sort("timestamp")
        self._length = len(self._df)

    @property
    def length(self) -> int:
        return self._length

    @property
    def start_time(self) -> datetime:
        return self._df["timestamp"][0]

    @property
    def end_time(self) -> datetime:
        return self._df["timestamp"][-1]

    def iter_ticks(self) -> Iterator[Tick]:
        """Iterate over ticks in chronological order."""
        for row in self._df.iter_rows(named=True):
            yield Tick(
                timestamp=row["timestamp"],
                price=Decimal(str(row["price"])),
                volume=int(row["volume"]),
                product=ProductType(row["product"]) if "product" in row else self._product,
                session=Session[row["session"]] if "session" in row else Session.DAY,
            )

    def slice(self, start: datetime, end: datetime) -> "DataFeed":
        """Return a new DataFeed filtered to a time range."""
        filtered = self._df.filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
        )
        return DataFeed(filtered, self._product)

    def to_dataframe(self) -> pl.DataFrame:
        """Return the underlying DataFrame."""
        return self._df
