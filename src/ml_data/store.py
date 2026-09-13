"""Connecting to the store and scanning only the partitions a query needs.

bear-lake lays a partitioned table out as `<store>/<table>/<year>/<symbol>.parquet`
with a `metadata.json` beside it. `bl.table()` globs the whole table, which
is right for the small tables and wrong for a per-symbol read of a 60 GB
chain table, so `scan` builds the path list from the partition keys in the
table's metadata and falls back to the glob otherwise.
"""

import datetime as dt
import json
import os
from pathlib import Path

import bear_lake as bl
import polars as pl
from dotenv import load_dotenv

load_dotenv()


def connect(path: str | Path | None = None) -> bl.Database:
    """Connect to the store at `path`, or at `ML_DATA_STORE`."""
    path = path or os.getenv("ML_DATA_STORE")
    if not path:
        raise RuntimeError("set ML_DATA_STORE to the bear-lake directory, or pass a path")
    return bl.connect(str(Path(path).expanduser()))


def store_path() -> Path:
    if not bl.CONNECTED:
        raise RuntimeError("not connected: call ml_data.connect() first")
    return Path(bl.DATABASE_PATH)


def table_dir(name: str) -> Path:
    return store_path() / name


def partition_keys(name: str) -> list[str]:
    metadata_path = table_dir(name) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"no table {name!r} in {store_path()}; run `uv run pipelines {name.replace('_', '-')}`"
            " in ml-data-pipelines"
        )
    return json.loads(metadata_path.read_text()).get("partition_keys") or []


def available_years(name: str) -> list[int]:
    directory = table_dir(name)
    if not directory.is_dir():
        return []
    return sorted(int(child.stem) for child in directory.iterdir() if child.stem.isdigit())


def available_symbols(name: str, year: int) -> list[str]:
    directory = table_dir(name) / str(year)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def scan(
    name: str,
    years: list[int] | None = None,
    symbols: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy scan of `name`, opening only the partitions for `years` and `symbols`."""
    keys = partition_keys(name)
    if keys == ["year", "symbol"] and (years is not None or symbols is not None):
        years = years if years is not None else available_years(name)
        paths = []
        for year in years:
            names = symbols if symbols is not None else available_symbols(name, year)
            paths.extend(
                table_dir(name) / str(year) / f"{symbol}.parquet"
                for symbol in names
                if (table_dir(name) / str(year) / f"{symbol}.parquet").exists()
            )
    elif keys == ["year"] and years is not None:
        paths = [
            table_dir(name) / f"{year}.parquet"
            for year in years
            if (table_dir(name) / f"{year}.parquet").exists()
        ]
    else:
        paths = sorted(table_dir(name).glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {name} partitions for years={years} symbols={symbols}")
    frame = pl.scan_parquet(paths)
    if symbols is not None and "symbol" in frame.collect_schema().names():
        frame = frame.filter(pl.col("symbol").is_in(symbols))
    return frame


def years_in(start: dt.date | None, end: dt.date | None, name: str) -> list[int] | None:
    """The years a window touches, clipped to what is on disk; None means every year."""
    if start is None and end is None:
        return None
    on_disk = available_years(name)
    first = start.year if start else (on_disk[0] if on_disk else None)
    last = end.year if end else (on_disk[-1] if on_disk else None)
    if first is None or last is None:
        return None
    return [year for year in on_disk if first <= year <= last]


def in_window(
    frame: pl.LazyFrame, start: dt.date | None, end: dt.date | None, column: str = "date"
) -> pl.LazyFrame:
    if start is not None:
        frame = frame.filter(pl.col(column) >= start)
    if end is not None:
        frame = frame.filter(pl.col(column) <= end)
    return frame


def describe() -> str:
    """One line per table: partitions and size."""
    lines = []
    for directory in sorted(store_path().iterdir()):
        if not (directory / "metadata.json").exists():
            continue
        files = list(directory.glob("**/*.parquet"))
        size = sum(file.stat().st_size for file in files)
        lines.append(f"{directory.name:<20} {len(files):>5} files  {size / 1e9:7.2f} GB")
    return "\n".join(lines)
