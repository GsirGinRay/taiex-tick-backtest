"""Tests for TAIFEX data parser."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from taiex_backtest.data.taifex_parser import (
    _parse_timestamp,
    _determine_session,
    _find_column_indices,
    parse_taifex_csv,
    parse_taifex_to_dataframe,
    TAIFEX_PRODUCT_MAP,
)
from taiex_backtest.domain.enums import ProductType, Session


class TestParseTimestamp:
    def test_basic_time(self):
        ts = _parse_timestamp("20240102", "090000")
        assert ts == datetime(2024, 1, 2, 9, 0, 0)

    def test_with_microseconds(self):
        ts = _parse_timestamp("20240102", "090000123456")
        assert ts == datetime(2024, 1, 2, 9, 0, 0, 123456)

    def test_with_partial_microseconds(self):
        ts = _parse_timestamp("20240102", "09000012")
        assert ts == datetime(2024, 1, 2, 9, 0, 0, 120000)

    def test_end_of_day(self):
        ts = _parse_timestamp("20240102", "134500")
        assert ts == datetime(2024, 1, 2, 13, 45, 0)

    def test_night_session(self):
        ts = _parse_timestamp("20240102", "150000")
        assert ts == datetime(2024, 1, 2, 15, 0, 0)

    def test_midnight(self):
        ts = _parse_timestamp("20240103", "010000")
        assert ts == datetime(2024, 1, 3, 1, 0, 0)

    def test_strips_whitespace(self):
        ts = _parse_timestamp("20240102", "  090000  ")
        assert ts == datetime(2024, 1, 2, 9, 0, 0)


class TestDetermineSession:
    def test_day_session_start(self):
        ts = datetime(2024, 1, 2, 8, 45, 0)
        assert _determine_session(ts) == Session.DAY

    def test_day_session_mid(self):
        ts = datetime(2024, 1, 2, 10, 30, 0)
        assert _determine_session(ts) == Session.DAY

    def test_day_session_end(self):
        ts = datetime(2024, 1, 2, 13, 45, 0)
        assert _determine_session(ts) == Session.DAY

    def test_night_session_start(self):
        ts = datetime(2024, 1, 2, 15, 0, 0)
        assert _determine_session(ts) == Session.NIGHT

    def test_night_session_late(self):
        ts = datetime(2024, 1, 2, 22, 0, 0)
        assert _determine_session(ts) == Session.NIGHT

    def test_early_morning_night(self):
        ts = datetime(2024, 1, 3, 3, 0, 0)
        assert _determine_session(ts) == Session.NIGHT

    def test_before_day_session(self):
        ts = datetime(2024, 1, 2, 8, 0, 0)
        assert _determine_session(ts) == Session.NIGHT

    def test_between_sessions(self):
        ts = datetime(2024, 1, 2, 14, 0, 0)
        assert _determine_session(ts) == Session.NIGHT

    def test_day_session_nine_am(self):
        ts = datetime(2024, 1, 2, 9, 0, 0)
        assert _determine_session(ts) == Session.DAY

    def test_day_session_noon(self):
        ts = datetime(2024, 1, 2, 12, 0, 0)
        assert _determine_session(ts) == Session.DAY


class TestFindColumnIndices:
    def test_chinese_headers(self):
        columns = [
            "交易日期", "商品代號", "到期月份(週別)",
            "成交時間", "成交價格", "成交數量(B+S)",
        ]
        result = _find_column_indices(columns)
        assert result is not None
        assert result["date"] == 0
        assert result["product"] == 1
        assert result["expiry"] == 2
        assert result["time"] == 3
        assert result["price"] == 4
        assert result["volume"] == 5

    def test_missing_required_column(self):
        columns = ["交易日期", "商品代號"]
        result = _find_column_indices(columns)
        assert result is None

    def test_extra_columns(self):
        columns = [
            "交易日期", "商品代號", "到期月份(週別)",
            "成交時間", "成交價格", "成交數量(B+S)",
            "近月價格", "遠月價格",
        ]
        result = _find_column_indices(columns)
        assert result is not None

    def test_english_headers(self):
        columns = [
            "trade_date", "product_code", "expiry",
            "trade_time", "trade_price", "volume",
        ]
        result = _find_column_indices(columns)
        assert result is not None
        assert result["date"] == 0
        assert result["product"] == 1
        assert result["time"] == 3
        assert result["price"] == 4
        assert result["volume"] == 5

    def test_missing_volume_returns_none(self):
        columns = ["交易日期", "商品代號", "到期月份(週別)", "成交時間", "成交價格"]
        result = _find_column_indices(columns)
        assert result is None

    def test_columns_with_spaces(self):
        columns = [" 交易日期 ", " 商品代號 ", "到期月份(週別)", "成交時間", "成交價格", "成交數量(B+S)"]
        result = _find_column_indices(columns)
        assert result is not None


class TestProductMap:
    def test_tx_mapping(self):
        assert TAIFEX_PRODUCT_MAP["TX"] == ProductType.TX

    def test_mtx_mapping(self):
        assert TAIFEX_PRODUCT_MAP["MTX"] == ProductType.MTX

    def test_xmt_mapping(self):
        assert TAIFEX_PRODUCT_MAP["XMT"] == ProductType.XMT

    def test_all_products_present(self):
        assert set(TAIFEX_PRODUCT_MAP.keys()) == {"TX", "MTX", "XMT"}


class TestParseTaifexCsv:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample TAIFEX-format CSV file."""
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,2,,\n"
            "20240102,TX  ,202401,090001,20005,1,,\n"
            "20240102,TX  ,202401,090002,19995,3,,\n"
            "20240102,MTX ,202401,090000,20000,5,,\n"
            "20240102,TX  ,202401,150000,20010,1,,\n"
        )
        filepath = tmp_path / "test_taifex.csv"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    @pytest.fixture
    def multi_expiry_csv(self, tmp_path: Path) -> Path:
        """CSV with multiple expiry months."""
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,2,,\n"
            "20240102,TX  ,202402,090001,20050,1,,\n"
            "20240102,TX  ,202401,090002,20010,3,,\n"
        )
        filepath = tmp_path / "multi_expiry.csv"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def test_parse_basic(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, encoding="utf-8")
        assert len(ticks) == 5

    def test_parse_filter_product(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, products=["TX"], encoding="utf-8")
        assert all(t.product == ProductType.TX for t in ticks)
        assert len(ticks) == 4

    def test_parse_prices_decimal(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, encoding="utf-8")
        assert all(isinstance(t.price, Decimal) for t in ticks)

    def test_parse_sorted_by_timestamp(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, encoding="utf-8")
        for i in range(1, len(ticks)):
            assert ticks[i].timestamp >= ticks[i - 1].timestamp

    def test_parse_session_detection(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, products=["TX"], encoding="utf-8")
        day_ticks = [t for t in ticks if t.session == Session.DAY]
        night_ticks = [t for t in ticks if t.session == Session.NIGHT]
        assert len(day_ticks) == 3
        assert len(night_ticks) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_taifex_csv(Path("nonexistent.csv"))

    def test_parse_volumes_positive(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, encoding="utf-8")
        assert all(t.volume > 0 for t in ticks)

    def test_parse_empty_file(self, tmp_path: Path):
        filepath = tmp_path / "empty.csv"
        filepath.write_text(
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n",
            encoding="utf-8",
        )
        ticks = parse_taifex_csv(filepath, encoding="utf-8")
        assert len(ticks) == 0

    def test_parse_invalid_lines_skipped(self, tmp_path: Path):
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,2,,\n"
            "bad,line\n"
            "20240102,TX  ,202401,090001,abc,1,,\n"
            "20240102,TX  ,202401,090002,20010,3,,\n"
        )
        filepath = tmp_path / "with_errors.csv"
        filepath.write_text(content, encoding="utf-8")
        ticks = parse_taifex_csv(filepath, encoding="utf-8")
        assert len(ticks) == 2

    def test_parse_mtx_product(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, products=["MTX"], encoding="utf-8")
        assert len(ticks) == 1
        assert ticks[0].product == ProductType.MTX
        assert ticks[0].price == Decimal("20000")
        assert ticks[0].volume == 5

    def test_near_month_only_filters_far_expiry(self, multi_expiry_csv: Path):
        ticks = parse_taifex_csv(
            multi_expiry_csv, products=["TX"], encoding="utf-8", near_month_only=True,
        )
        assert len(ticks) == 2
        assert all(t.price != Decimal("20050") for t in ticks)

    def test_near_month_only_false_includes_all(self, multi_expiry_csv: Path):
        ticks = parse_taifex_csv(
            multi_expiry_csv, products=["TX"], encoding="utf-8", near_month_only=False,
        )
        assert len(ticks) == 3

    def test_unknown_product_skipped(self, tmp_path: Path):
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,ZZZ ,202401,090000,20000,2,,\n"
            "20240102,TX  ,202401,090001,20005,1,,\n"
        )
        filepath = tmp_path / "unknown_product.csv"
        filepath.write_text(content, encoding="utf-8")
        ticks = parse_taifex_csv(filepath, encoding="utf-8")
        assert len(ticks) == 1
        assert ticks[0].product == ProductType.TX

    def test_zero_volume_skipped(self, tmp_path: Path):
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,0,,\n"
            "20240102,TX  ,202401,090001,20005,1,,\n"
        )
        filepath = tmp_path / "zero_vol.csv"
        filepath.write_text(content, encoding="utf-8")
        ticks = parse_taifex_csv(filepath, encoding="utf-8")
        assert len(ticks) == 1

    def test_zero_price_skipped(self, tmp_path: Path):
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,0,2,,\n"
            "20240102,TX  ,202401,090001,20005,1,,\n"
        )
        filepath = tmp_path / "zero_price.csv"
        filepath.write_text(content, encoding="utf-8")
        ticks = parse_taifex_csv(filepath, encoding="utf-8")
        assert len(ticks) == 1

    def test_tick_attributes(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, products=["TX"], encoding="utf-8")
        first_tick = ticks[0]
        assert first_tick.timestamp == datetime(2024, 1, 2, 9, 0, 0)
        assert first_tick.price == Decimal("20000")
        assert first_tick.volume == 2
        assert first_tick.product == ProductType.TX
        assert first_tick.session == Session.DAY

    def test_invalid_header_raises(self, tmp_path: Path):
        content = "col_a,col_b,col_c\n1,2,3\n"
        filepath = tmp_path / "bad_header.csv"
        filepath.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Cannot parse TAIFEX CSV header"):
            parse_taifex_csv(filepath, encoding="utf-8")

    def test_default_products_includes_all(self, sample_csv: Path):
        ticks = parse_taifex_csv(sample_csv, products=None, encoding="utf-8")
        product_types = {t.product for t in ticks}
        assert ProductType.TX in product_types
        assert ProductType.MTX in product_types

    def test_string_filepath_accepted(self, sample_csv: Path):
        ticks = parse_taifex_csv(str(sample_csv), encoding="utf-8")
        assert len(ticks) == 5


class TestParseTaifexToDataframe:
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,2,,\n"
            "20240102,TX  ,202401,090001,20005,1,,\n"
            "20240102,TX  ,202401,090002,19995,3,,\n"
        )
        filepath = tmp_path / "test_df.csv"
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def test_returns_dataframe(self, sample_csv: Path):
        import polars as pl

        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3

    def test_schema_columns(self, sample_csv: Path):
        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert "timestamp" in df.columns
        assert "price" in df.columns
        assert "volume" in df.columns
        assert "product" in df.columns
        assert "session" in df.columns

    def test_empty_csv(self, tmp_path: Path):
        import polars as pl

        filepath = tmp_path / "empty.csv"
        filepath.write_text(
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n",
            encoding="utf-8",
        )
        df = parse_taifex_to_dataframe(filepath, encoding="utf-8")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0

    def test_product_column_values(self, sample_csv: Path):
        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert all(v == "TX" for v in df["product"].to_list())

    def test_session_column_values(self, sample_csv: Path):
        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert all(v == "DAY" for v in df["session"].to_list())

    def test_price_as_float(self, sample_csv: Path):
        import polars as pl

        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert df["price"].dtype == pl.Float64

    def test_volume_as_int(self, sample_csv: Path):
        import polars as pl

        df = parse_taifex_to_dataframe(sample_csv, encoding="utf-8")
        assert df["volume"].dtype == pl.Int32

    def test_filter_by_product(self, tmp_path: Path):
        import polars as pl

        content = (
            "交易日期,商品代號,到期月份(週別),成交時間,成交價格,成交數量(B+S),近月價格,遠月價格\n"
            "20240102,TX  ,202401,090000,20000,2,,\n"
            "20240102,MTX ,202401,090000,20000,5,,\n"
        )
        filepath = tmp_path / "mixed.csv"
        filepath.write_text(content, encoding="utf-8")
        df = parse_taifex_to_dataframe(filepath, products=["MTX"], encoding="utf-8")
        assert len(df) == 1
        assert df["product"][0] == "MTX"
