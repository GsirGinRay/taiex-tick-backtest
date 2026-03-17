"""Parser for TAIFEX (台灣期交所) tick-by-tick transaction data."""

from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator

import polars as pl

from ..domain.enums import ProductType, Session
from ..domain.models import Tick
from .schema import TICK_SCHEMA


# TAIFEX product code mapping
TAIFEX_PRODUCT_MAP: dict[str, ProductType] = {
    "TX": ProductType.TX,
    "MTX": ProductType.MTX,
    "XMT": ProductType.XMT,
}

# Column name mapping (Chinese -> English)
TAIFEX_COLUMNS = {
    "交易日期": "trade_date",
    "商品代號": "product_code",
    "到期月份(週別)": "expiry",
    "成交時間": "trade_time",
    "成交價格": "trade_price",
    "成交數量(B+S)": "volume",
    "近月價格": "near_price",
    "遠月價格": "far_price",
}


def _determine_session(t: datetime) -> Session:
    """Determine trading session from timestamp."""
    hour = t.hour
    minute = t.minute
    # Day session: 08:45 - 13:45
    if (hour == 8 and minute >= 45) or (9 <= hour <= 13) or (hour == 13 and minute <= 45):
        return Session.DAY
    return Session.NIGHT


def _parse_timestamp(date_str: str, time_str: str) -> datetime:
    """Parse TAIFEX date and time strings into datetime.

    Args:
        date_str: Date in YYYYMMDD format.
        time_str: Time in HHMMSS or HHMMSSffffff format.
    """
    time_str = time_str.strip()
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    microsecond = 0
    if len(time_str) > 6:
        micro_str = time_str[6:]
        microsecond = int(micro_str.ljust(6, "0")[:6])

    return datetime(year, month, day, hour, minute, second, microsecond)


def parse_taifex_csv(
    filepath: Path | str,
    products: list[str] | None = None,
    encoding: str = "big5",
    near_month_only: bool = True,
) -> list[Tick]:
    """Parse a TAIFEX CSV file into Tick objects.

    Args:
        filepath: Path to the CSV file.
        products: List of product codes to include (e.g., ["TX", "MTX"]).
                  Defaults to all supported products.
        encoding: File encoding (default: big5 for TAIFEX files).
        near_month_only: If True, only include nearest expiry month.

    Returns:
        List of Tick objects sorted by timestamp.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TAIFEX CSV file not found: {filepath}")

    if products is None:
        products = list(TAIFEX_PRODUCT_MAP.keys())

    ticks: list[Tick] = []

    with open(filepath, encoding=encoding, errors="replace") as f:
        header = f.readline().strip()
        # Parse header to find column indices
        columns = [c.strip() for c in header.split(",")]
        col_indices = _find_column_indices(columns)

        if col_indices is None:
            raise ValueError(f"Cannot parse TAIFEX CSV header: {header}")

        # Track nearest expiry per product for filtering
        expiry_map: dict[str, str] = {}

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            try:
                product_code = parts[col_indices["product"]].strip()
                if product_code not in products:
                    continue

                if product_code not in TAIFEX_PRODUCT_MAP:
                    continue

                date_str = parts[col_indices["date"]].strip()
                time_str = parts[col_indices["time"]].strip()
                price_str = parts[col_indices["price"]].strip()
                volume_str = parts[col_indices["volume"]].strip()
                expiry = parts[col_indices["expiry"]].strip() if "expiry" in col_indices else ""

                if near_month_only and expiry:
                    if product_code not in expiry_map:
                        expiry_map[product_code] = expiry
                    elif expiry != expiry_map[product_code]:
                        # Compare expiry months, keep the nearest
                        if expiry < expiry_map[product_code]:
                            expiry_map[product_code] = expiry
                        else:
                            continue

                timestamp = _parse_timestamp(date_str, time_str)
                price = Decimal(price_str)
                volume = int(volume_str)

                if volume <= 0 or price <= 0:
                    continue

                product_type = TAIFEX_PRODUCT_MAP[product_code]
                session = _determine_session(timestamp)

                ticks.append(Tick(
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    product=product_type,
                    session=session,
                ))
            except (ValueError, InvalidOperation, IndexError):
                continue

    # If near_month_only, do a second pass to filter
    if near_month_only and expiry_map:
        # Already filtered during parsing for first-seen expiry
        pass

    ticks.sort(key=lambda t: t.timestamp)
    return ticks


def _find_column_indices(columns: list[str]) -> dict[str, int] | None:
    """Find column indices from header, supporting both Chinese and English names."""
    indices: dict[str, int] = {}

    for i, col in enumerate(columns):
        col_clean = col.strip().replace(" ", "")
        if col_clean in ("交易日期", "trade_date"):
            indices["date"] = i
        elif col_clean in ("商品代號", "product_code"):
            indices["product"] = i
        elif "到期月份" in col_clean or col_clean == "expiry":
            indices["expiry"] = i
        elif col_clean in ("成交時間", "trade_time"):
            indices["time"] = i
        elif col_clean in ("成交價格", "trade_price"):
            indices["price"] = i
        elif "成交數量" in col_clean or col_clean == "volume":
            indices["volume"] = i

    required = {"date", "product", "time", "price", "volume"}
    if required.issubset(indices.keys()):
        return indices
    return None


def parse_taifex_to_dataframe(
    filepath: Path | str,
    products: list[str] | None = None,
    encoding: str = "big5",
) -> pl.DataFrame:
    """Parse TAIFEX CSV directly into a Polars DataFrame.

    Args:
        filepath: Path to the CSV file.
        products: Product codes to include.
        encoding: File encoding.

    Returns:
        Polars DataFrame matching TICK_SCHEMA.
    """
    ticks = parse_taifex_csv(filepath, products=products, encoding=encoding)

    if not ticks:
        return pl.DataFrame(schema=TICK_SCHEMA)

    return pl.DataFrame({
        "timestamp": [t.timestamp for t in ticks],
        "price": [float(t.price) for t in ticks],
        "volume": [t.volume for t in ticks],
        "product": [t.product.value for t in ticks],
        "session": [t.session.name for t in ticks],
    }).cast(TICK_SCHEMA)
