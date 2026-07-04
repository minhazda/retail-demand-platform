"""Raw-data ingestion: UCI Online Retail II -> DuckDB ``raw.transactions``.

Primary source is the CRAN-package GitHub mirror (a small .rda, stable and
fast); the UCI zip is the fallback. Column names are normalised to snake_case
so the dbt layer has a single contract regardless of source.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

log = logging.getLogger(__name__)

MIRROR_URL = "https://raw.githubusercontent.com/allanvc/onlineretail2/master/data/onlineretail2.rda"
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

RENAMES = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_ts",
    "Price": "price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Customer_ID": "customer_id",
    "Country": "country",
}
REQUIRED = ("invoice", "stock_code", "quantity", "invoice_ts", "price")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to the raw contract and coerce types; raises on missing columns."""
    out = df.rename(columns=RENAMES)
    missing = set(REQUIRED) - set(out.columns)
    if missing:
        raise ValueError(f"raw schema mismatch, missing: {sorted(missing)}")
    out["invoice"] = out["invoice"].astype(str)
    out["stock_code"] = out["stock_code"].astype(str)
    out["invoice_ts"] = pd.to_datetime(out["invoice_ts"])
    return out


def fetch_raw(cache_path: Path) -> pd.DataFrame:
    """Download (or reuse a cached copy of) the raw dataset."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        log.info("using cached raw data: %s", cache_path)
        return pd.read_parquet(cache_path)
    try:
        df = _fetch_mirror()
    except Exception as exc:  # noqa: BLE001 - any mirror failure falls back to UCI
        log.warning("mirror fetch failed (%s); falling back to UCI", exc)
        df = _fetch_uci()
    df = normalise(df)
    df.to_parquet(cache_path, index=False)
    return df


def _fetch_mirror() -> pd.DataFrame:
    import tempfile

    import pyreadr

    response = requests.get(MIRROR_URL, timeout=120)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as handle:
        handle.write(response.content)
        temp_path = handle.name
    result = pyreadr.read_r(temp_path)
    return next(iter(result.values()))


def _fetch_uci() -> pd.DataFrame:
    response = requests.get(UCI_URL, timeout=300)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
        with archive.open(name) as handle:
            sheets = pd.read_excel(handle, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def load_to_duckdb(df: pd.DataFrame, duckdb_path: Path) -> int:
    """Replace ``raw.transactions`` with ``df``; returns the row count."""
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(duckdb_path)) as conn:
        conn.execute("create schema if not exists raw")
        conn.register("df", df)
        conn.execute("create or replace table raw.transactions as select * from df")
        count = conn.execute("select count(*) from raw.transactions").fetchone()
    return int(count[0]) if count else 0
