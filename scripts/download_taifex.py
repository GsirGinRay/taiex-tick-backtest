"""Download TAIFEX (台灣期交所) tick-by-tick transaction data.

Usage:
    python download_taifex.py --date 2024-01-02
    python download_taifex.py --start 2024-01-02 --end 2024-01-31
    python download_taifex.py --date 2024-01-02 --product TX
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import URLError, HTTPError

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

DEFAULT_OUTPUT_DIR = project_root / "data" / "raw"
TAIFEX_URL = "https://www.taifex.com.tw/cht/3/futDataDown"


def download_day(
    trade_date: date,
    output_dir: Path,
    product: str = "TX",
    delay: float = 2.0,
) -> Path | None:
    """Download tick data for a single trading day.

    Args:
        trade_date: The trading date.
        output_dir: Directory to save the CSV file.
        product: Product code (TX, MTX, XMT).
        delay: Delay in seconds between requests (be polite).

    Returns:
        Path to the downloaded file, or None if download failed.
    """
    date_str = trade_date.strftime("%Y/%m/%d")
    commodity_map = {
        "TX": "TX",
        "MTX": "MTX",
        "XMT": "XMT",
    }

    commodity_id = commodity_map.get(product, product)

    form_data = {
        "down_type": "1",
        "commodity_id": commodity_id,
        "queryStartDate": date_str,
        "queryEndDate": date_str,
    }

    encoded = urlencode(form_data).encode("utf-8")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.taifex.com.tw/cht/3/dlFutDataDown",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"taifex_{product}_{trade_date.strftime('%Y%m%d')}.csv"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  [skip] Already exists: {filename}")
        return output_path

    try:
        req = Request(TAIFEX_URL, data=encoded, headers=headers, method="POST")
        with urlopen(req, timeout=30) as response:
            content = response.read()

            if len(content) < 100:
                print(f"  [warn] Empty or invalid response for {trade_date} (likely a holiday)")
                return None

            output_path.write_bytes(content)
            print(f"  [done] {filename} ({len(content):,} bytes)")
            return output_path

    except HTTPError as e:
        print(f"  [error] HTTP {e.code} for {trade_date}: {e.reason}")
        return None
    except URLError as e:
        print(f"  [error] URL error for {trade_date}: {e.reason}")
        return None
    except Exception as e:
        print(f"  [error] Failed for {trade_date}: {e}")
        return None
    finally:
        time.sleep(delay)


def daterange(start: date, end: date):
    """Generate dates from start to end (inclusive), skipping weekends."""
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Friday=4
            yield current
        current += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Download TAIFEX tick-by-tick transaction data"
    )
    parser.add_argument(
        "--date", type=str, help="Single date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--start", type=str, help="Start date for range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, help="End date for range (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--product", type=str, default="TX",
        choices=["TX", "MTX", "XMT"],
        help="Product code (default: TX)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between requests in seconds (default: 2.0)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.date:
        dates = [datetime.strptime(args.date, "%Y-%m-%d").date()]
    elif args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
        dates = list(daterange(start, end))
    else:
        parser.error("Specify --date or both --start and --end")
        return

    print(f"Downloading TAIFEX {args.product} data for {len(dates)} trading day(s)...")
    print(f"Output directory: {output_dir}")
    print()

    success = 0
    for d in dates:
        result = download_day(d, output_dir, args.product, args.delay)
        if result is not None:
            success += 1

    print(f"\nDone: {success}/{len(dates)} files downloaded.")


if __name__ == "__main__":
    main()
