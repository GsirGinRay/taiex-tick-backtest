"""Tests for data layer: synthetic generation, schema, writer, feed."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from taiex_backtest.data.schema import TICK_SCHEMA, validate_schema
from taiex_backtest.data.synthetic import generate_ticks
from taiex_backtest.data.writer import ticks_to_dataframe, write_parquet, write_ticks
from taiex_backtest.data.feed import DataFeed
from taiex_backtest.domain.enums import ProductType, Session


class TestSchema:
    def test_tick_schema_columns(self):
        assert "timestamp" in TICK_SCHEMA
        assert "price" in TICK_SCHEMA
        assert "volume" in TICK_SCHEMA

    def test_validate_schema_valid(self):
        df = pl.DataFrame({
            "timestamp": [datetime.now()],
            "price": [20000.0],
            "volume": [1],
            "product": ["TX"],
            "session": ["DAY"],
        }).cast(TICK_SCHEMA)
        assert validate_schema(df)

    def test_validate_schema_missing_column(self):
        df = pl.DataFrame({"timestamp": [datetime.now()], "price": [20000.0]})
        assert not validate_schema(df)


class TestSynthetic:
    def test_generate_ticks_default(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        assert len(ticks) == 100
        assert all(t.product == ProductType.TX for t in ticks)

    def test_generate_ticks_deterministic(self):
        ticks1 = generate_ticks(num_ticks=50, seed=123)
        ticks2 = generate_ticks(num_ticks=50, seed=123)
        assert [t.price for t in ticks1] == [t.price for t in ticks2]

    def test_generate_ticks_prices_integer(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        for tick in ticks:
            assert tick.price == tick.price.to_integral_value()

    def test_generate_ticks_timestamps_ascending(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        for i in range(1, len(ticks)):
            assert ticks[i].timestamp > ticks[i - 1].timestamp

    def test_generate_ticks_volumes_positive(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        assert all(t.volume > 0 for t in ticks)

    def test_generate_ticks_custom_product(self):
        ticks = generate_ticks(num_ticks=10, product=ProductType.MTX, seed=42)
        assert all(t.product == ProductType.MTX for t in ticks)


class TestWriter:
    def test_ticks_to_dataframe(self):
        ticks = generate_ticks(num_ticks=50, seed=42)
        df = ticks_to_dataframe(ticks)
        assert len(df) == 50
        assert validate_schema(df)

    def test_ticks_to_dataframe_empty(self):
        df = ticks_to_dataframe([])
        assert len(df) == 0

    def test_write_parquet(self, tmp_path: Path):
        ticks = generate_ticks(num_ticks=100, seed=42)
        df = ticks_to_dataframe(ticks)
        path = tmp_path / "test.parquet"
        result = write_parquet(df, path)
        assert result.exists()
        loaded = pl.read_parquet(path)
        assert len(loaded) == 100

    def test_write_ticks(self, tmp_path: Path):
        ticks = generate_ticks(num_ticks=100, seed=42)
        path = tmp_path / "sub" / "test.parquet"
        result = write_ticks(ticks, path)
        assert result.exists()


class TestDataFeed:
    def test_feed_from_dataframe(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)
        assert feed.length == 100

    def test_feed_iter_ticks(self):
        ticks = generate_ticks(num_ticks=50, seed=42)
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)
        result = list(feed.iter_ticks())
        assert len(result) == 50
        assert result[0].price == ticks[0].price

    def test_feed_from_parquet(self, tmp_path: Path):
        ticks = generate_ticks(num_ticks=100, seed=42)
        path = write_ticks(ticks, tmp_path / "test.parquet")
        feed = DataFeed(path)
        assert feed.length == 100

    def test_feed_start_end_time(self):
        ticks = generate_ticks(num_ticks=100, seed=42)
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)
        assert feed.start_time < feed.end_time

    def test_feed_slice(self):
        ticks = generate_ticks(num_ticks=1000, seed=42)
        df = ticks_to_dataframe(ticks)
        feed = DataFeed(df)
        start = ticks[100].timestamp
        end = ticks[200].timestamp
        sliced = feed.slice(start, end)
        assert sliced.length <= 101
        assert sliced.length > 0
