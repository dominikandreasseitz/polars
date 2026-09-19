"""Benchmark: `scan_iceberg().filter(pl_expr)` vs `scan_iceberg(row_filter=...)`.

Uses a local sqlite PyIceberg catalog and filesystem warehouse, a table with
many small files, and a non-trivial row filter (nested AND/OR + a wide
`is_in`) to stress the polars -> pyarrow -> PyIceberg conversion bridge that
the `.filter()` path relies on.

Reports median / p95 wall-clock latency over repeated collects for both
paths. Not a pytest module - run directly: `python bench_iceberg_row_filter.py`.
Requires `pyiceberg`, `pyarrow`, `numpy`.
"""

from __future__ import annotations

import shutil
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.table import Table

T = TypeVar("T")

WAREHOUSE = Path("/tmp/iceberg_bench_warehouse")
CATALOG_DB = Path("/tmp/iceberg_bench_catalog.db")

N_FILES = 60
ROWS_PER_FILE = 20_000
N_TRIALS = 25
CATEGORIES = [f"cat_{i}" for i in range(20)]
# The categories our filter will select - a non-trivial subset, not a
# single-partition equality.
SELECTED_CATEGORIES = [f"cat_{i}" for i in range(0, 20, 3)]  # 7 of 20
ID_THRESHOLD = int(N_FILES * ROWS_PER_FILE * 0.3)


def build_table() -> Table:
    shutil.rmtree(WAREHOUSE, ignore_errors=True)
    CATALOG_DB.unlink(missing_ok=True)
    WAREHOUSE.mkdir(parents=True)

    import numpy as np
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "bench",
        uri=f"sqlite:///{CATALOG_DB}",
        warehouse=f"file://{WAREHOUSE}",
    )
    catalog.create_namespace("ns")

    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("category", pa.string()),
            ("value", pa.float64()),
            ("ts", pa.timestamp("us")),
        ]
    )

    tbl = catalog.create_table("ns.events", schema=schema)

    rng = np.random.default_rng(42)
    for i in range(N_FILES):
        cats = rng.choice(CATEGORIES, size=ROWS_PER_FILE)
        batch = pa.table(
            {
                "id": pa.array(
                    np.arange(i * ROWS_PER_FILE, (i + 1) * ROWS_PER_FILE), type=pa.int64()
                ),
                "category": pa.array(cats, type=pa.string()),
                "value": pa.array(rng.normal(size=ROWS_PER_FILE), type=pa.float64()),
                "ts": pa.array(
                    np.full(ROWS_PER_FILE, i, dtype="datetime64[us]"), type=pa.timestamp("us")
                ),
            },
            schema=schema,
        )
        tbl.append(batch)

    return tbl


def polars_filter_expr() -> pl.Expr:
    return (
        pl.col("category").is_in(SELECTED_CATEGORIES)
        & (pl.col("value") > 0.5)
        & ((pl.col("id") < ID_THRESHOLD) | (pl.col("value") < -1.5))
    )


def pyiceberg_filter_expr() -> BooleanExpression:
    from pyiceberg.expressions import And, GreaterThan, In, LessThan, Or

    return And(
        In("category", SELECTED_CATEGORIES),
        GreaterThan("value", 0.5),
        Or(LessThan("id", ID_THRESHOLD), LessThan("value", -1.5)),
    )


def time_trials(fn: Callable[[], T], n: int = N_TRIALS) -> tuple[list[float], T]:
    result = fn()  # one warmup (metadata caching, page cache warmup)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return times, result


def summarize(label: str, times: list[float], n_rows: int) -> tuple[float, float]:
    times_sorted = sorted(times)
    p50 = statistics.median(times_sorted)
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    print(
        f"{label:32s}  rows={n_rows:>8d}  "
        f"median={p50 * 1000:8.2f}ms  p95={p95 * 1000:8.2f}ms  "
        f"min={min(times) * 1000:8.2f}ms  max={max(times) * 1000:8.2f}ms"
    )
    return p50, p95


def main() -> None:
    tbl = build_table()
    table_uri = tbl.metadata_location

    def run_polars_filter() -> int:
        lf = pl.scan_iceberg(table_uri).filter(polars_filter_expr())
        return len(lf.collect())

    def run_row_filter() -> int:
        lf = pl.scan_iceberg(table_uri, row_filter=pyiceberg_filter_expr())
        return len(lf.collect())

    def run_row_filter_plus_polars_filter() -> int:
        # sanity: same predicate expressed both ways should match row counts
        lf = pl.scan_iceberg(table_uri, row_filter=pyiceberg_filter_expr()).filter(
            polars_filter_expr()
        )
        return len(lf.collect())

    times_a, rows_a = time_trials(run_polars_filter)
    times_b, rows_b = time_trials(run_row_filter)
    times_c, rows_c = time_trials(run_row_filter_plus_polars_filter)

    assert rows_a == rows_b == rows_c, (
        f"row count mismatch: filter={rows_a} row_filter={rows_b} both={rows_c}"
    )

    print(f"\nTable: {N_FILES} files x {ROWS_PER_FILE} rows = {N_FILES * ROWS_PER_FILE} total rows")
    print(f"Trials per method: {N_TRIALS} (+1 warmup, discarded)\n")

    p50_a, p95_a = summarize(".filter(pl_expr)  [current]", times_a, rows_a)
    p50_b, p95_b = summarize("row_filter=pyiceberg_expr [new]", times_b, rows_b)
    summarize("row_filter + .filter() [both]", times_c, rows_c)

    print()
    print(f"p50 speedup (row_filter vs .filter): {p50_a / p50_b:.2f}x")
    print(f"p95 speedup (row_filter vs .filter): {p95_a / p95_b:.2f}x")

    if p95_b <= p95_a * 1.05:
        print("\nRESULT: row_filter is on par with or faster than .filter() at p95.")
    else:
        print("\nRESULT: row_filter is SLOWER than .filter() at p95 - investigate.")


if __name__ == "__main__":
    main()
