"""Convert TAIFEX CSV files to Parquet format.

Usage:
    python convert_csv_to_parquet.py data/raw/taifex_TX_20240102.csv
    python convert_csv_to_parquet.py data/raw/ --output data/processed/
    python convert_csv_to_parquet.py data/raw/ --merge --output data/processed/tx_2024.parquet
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import polars as pl

from taiex_backtest.data.taifex_parser import parse_taifex_to_dataframe
from taiex_backtest.data.writer import write_parquet
from taiex_backtest.data.schema import validate_schema


def convert_single(
    csv_path: Path,
    output_dir: Path,
    encoding: str = "big5",
    products: list[str] | None = None,
) -> Path | None:
    """Convert a single CSV file to Parquet."""
    try:
        df = parse_taifex_to_dataframe(csv_path, products=products, encoding=encoding)
        if len(df) == 0:
            print(f"  [skip] No valid data in {csv_path.name}")
            return None

        output_path = output_dir / csv_path.with_suffix(".parquet").name
        write_parquet(df, output_path)
        print(f"  [done] {csv_path.name} -> {output_path.name} ({len(df):,} ticks)")
        return output_path

    except Exception as e:
        print(f"  [error] {csv_path.name}: {e}")
        return None


def convert_and_merge(
    csv_files: list[Path],
    output_path: Path,
    encoding: str = "big5",
    products: list[str] | None = None,
) -> Path | None:
    """Convert multiple CSV files and merge into one Parquet."""
    frames: list[pl.DataFrame] = []

    for csv_path in sorted(csv_files):
        try:
            df = parse_taifex_to_dataframe(csv_path, products=products, encoding=encoding)
            if len(df) > 0:
                frames.append(df)
                print(f"  [parsed] {csv_path.name}: {len(df):,} ticks")
        except Exception as e:
            print(f"  [error] {csv_path.name}: {e}")

    if not frames:
        print("No valid data found.")
        return None

    merged = pl.concat(frames).sort("timestamp")
    write_parquet(merged, output_path)
    print(f"\n  [merged] {len(merged):,} ticks -> {output_path.name}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert TAIFEX CSV files to Parquet"
    )
    parser.add_argument(
        "input", type=str,
        help="CSV file or directory containing CSV files"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default=str(project_root / "data" / "processed"),
        help="Output directory or file path (for --merge)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all CSV files into a single Parquet file"
    )
    parser.add_argument(
        "--encoding", type=str, default="big5",
        help="CSV encoding (default: big5)"
    )
    parser.add_argument(
        "--products", type=str, nargs="+",
        default=None,
        help="Product codes to include (e.g., TX MTX)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        csv_files = [input_path]
    elif input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {input_path}")
            return
    else:
        print(f"Input not found: {input_path}")
        return

    print(f"Found {len(csv_files)} CSV file(s)")
    print()

    if args.merge:
        if not output_path.suffix:
            output_path = output_path / "merged.parquet"
        convert_and_merge(csv_files, output_path, args.encoding, args.products)
    else:
        output_dir = output_path if output_path.is_dir() or not output_path.suffix else output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        success = 0
        for f in csv_files:
            result = convert_single(f, output_dir, args.encoding, args.products)
            if result:
                success += 1
        print(f"\nDone: {success}/{len(csv_files)} files converted.")


if __name__ == "__main__":
    main()
