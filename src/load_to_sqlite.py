"""
Stream the accepted-loans CSV into a single SQLite table `loans`.

The file is ~2.2M rows x 151 cols, so we read it in chunks and append,
keeping memory bounded. SQLite is dynamically typed, so per-chunk dtype
drift is harmless for a raw landing table.

Usage:
    python src/load_to_sqlite.py
    python src/load_to_sqlite.py --chunksize 200000
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
DB_PATH = ROOT / "data" / "lending_club.db"
TABLE = "loans"


def load(chunksize: int) -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"ERROR: {CSV_PATH} not found.\n"
            "Run `python src/download_data.py` first, or drop the file in data/raw/."
        )

    if DB_PATH.exists():
        DB_PATH.unlink()  # rebuild from scratch so reruns are idempotent

    print(f"Loading {CSV_PATH.name} -> {DB_PATH.name} (table '{TABLE}')")
    t0 = time.time()
    total = 0
    with sqlite3.connect(DB_PATH) as con:
        reader = pd.read_csv(
            CSV_PATH,
            chunksize=chunksize,
            low_memory=False,
            compression="gzip",
        )
        for i, chunk in enumerate(reader):
            chunk.to_sql(
                TABLE, con, if_exists="replace" if i == 0 else "append", index=False
            )
            total += len(chunk)
            print(f"  chunk {i:>3}  rows so far: {total:>9,}", end="\r")

    dt = time.time() - t0
    print(f"\nDone: {total:,} rows in {dt:.0f}s -> {DB_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()
    load(args.chunksize)
