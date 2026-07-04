"""Normalisation contract and DuckDB load."""

from __future__ import annotations

import pandas as pd
import pytest

from retail_platform.ingest import load_to_duckdb, normalise


def test_normalise_renames_and_types(raw_frame: pd.DataFrame) -> None:
    out = normalise(raw_frame)
    assert {"invoice", "stock_code", "quantity", "invoice_ts", "price"} <= set(out.columns)
    assert out["invoice"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(out["invoice_ts"])


def test_normalise_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        normalise(pd.DataFrame({"Invoice": ["1"]}))


def test_load_to_duckdb_roundtrip(raw_frame: pd.DataFrame, tmp_path) -> None:
    import duckdb

    db = tmp_path / "wh.duckdb"
    rows = load_to_duckdb(normalise(raw_frame), db)
    assert rows == len(raw_frame)
    with duckdb.connect(str(db), read_only=True) as conn:
        got = conn.execute("select count(*) from raw.transactions").fetchone()
    assert got is not None and got[0] == rows
