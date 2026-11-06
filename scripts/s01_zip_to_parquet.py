"""Stream every measurement CSV in GFQA_v3.zip into a typed Parquet file.

The ZIP is read-only and never extracted. Each CSV is read in chunks with
Latin-1 (byte-reversible) decoding; text columns stay strings, Value is parsed
to float, and the Remark column is kept; the License URL column is dropped (it is
retained in the read-only ZIP).
"""
from __future__ import annotations

import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "inputs" / "GFQA_v3.zip"
OUT = ROOT / "data_processed" / "measurements"
OUT.mkdir(parents=True, exist_ok=True)

COLS = {
    "GEMS Station Number": "station",
    "Sample Date": "date",
    "Sample Time": "time",
    "Depth": "depth",
    "Parameter Code": "param",
    "Analysis Method Code": "method",
    "Value Flags": "flag",
    "Value": "value",
    "Unit": "unit",
    "Data Quality": "quality",
    "Integrated Value": "integrated",
    "Remark": "remark",
    "License Information": "license",
}


STR = pa.string()
SCHEMA = pa.schema([
    ("station", STR), ("date", pa.timestamp("ns")), ("time", STR), ("depth", pa.float64()),
    ("param", STR), ("method", STR), ("flag", STR), ("value", pa.float64()), ("unit", STR),
    ("quality", STR), ("integrated", STR), ("remark", STR), ("file", STR),
])


def convert(member: str) -> tuple[str, int]:
    target = OUT / member.replace(".csv", ".parquet")
    if target.exists():
        return member, pq.ParquetFile(target).metadata.num_rows
    writer = None
    n = 0
    with zipfile.ZipFile(ZIP) as z, z.open(member) as fh:
        reader = pd.read_csv(fh, encoding="latin-1", dtype=str, chunksize=400_000,
                             keep_default_na=False, na_values=[""])
        for chunk in reader:
            chunk = chunk.rename(columns=COLS)
            chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk["depth"] = pd.to_numeric(chunk["depth"], errors="coerce")
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk["file"] = member.replace(".csv", "")
            chunk = chunk.drop(columns=["license"])
            table = pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target.with_suffix(".tmp"), table.schema, compression="zstd")
            writer.write_table(table)
            n += len(chunk)
    writer.close()
    target.with_suffix(".tmp").rename(target)
    return member, n


if __name__ == "__main__":
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.endswith(".csv") and not m.startswith("GEMStat_")]
    members.sort(key=lambda m: -zipfile.ZipFile(ZIP).getinfo(m).file_size)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    total = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for member, n in ex.map(convert, members):
            total += n
            print(f"{member:45s} {n:>10,d}", flush=True)
    print(f"TOTAL {total:,d}")
