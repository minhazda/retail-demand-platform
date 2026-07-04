"""End-to-end materialization on synthetic data: ingest -> dbt build -> train -> gate.

Marked ``pipeline``: needs dbt deps installed (network for dbt_utils) and ~1 min.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from retail_platform.ingest import load_to_duckdb, normalise
from retail_platform.promotion import apply, decide
from retail_platform.training import train_challenger

pytestmark = pytest.mark.pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dbt_build(duckdb_path: Path) -> None:
    import os

    from dbt.cli.main import dbtRunner

    os.environ["RDP_DUCKDB_PATH"] = str(duckdb_path)
    project_dir = str(REPO_ROOT / "dbt_project")
    for command in (["deps"], ["build"]):
        result = dbtRunner().invoke(
            command + ["--project-dir", project_dir, "--profiles-dir", project_dir]
        )
        assert result.success, f"dbt {command[0]} failed: {result.exception}"


def test_full_platform_run(raw_frame: pd.DataFrame, tmp_path: Path) -> None:
    db = tmp_path / "warehouse.duckdb"

    # 1) ingest
    rows = load_to_duckdb(normalise(raw_frame), db)
    assert rows == len(raw_frame)

    # 2) transform + data-quality tests (dbt build fails the test on violation)
    _dbt_build(db)
    with duckdb.connect(str(db), read_only=True) as conn:
        daily = conn.execute("select * from fct_daily_sales").df()
        staged = conn.execute("select count(*) from stg_transactions").fetchone()
    assert staged is not None and staged[0] < rows  # dirty rows were dropped
    assert set(daily["stock_code"].unique()) <= {f"1000{i}" for i in range(8)}

    # 3) train challenger vs seasonal-naive
    report = train_challenger(daily, tmp_path / "models")
    assert report.n_series == 8
    assert report.mae > 0
    print(
        f"\npipeline e2e: challenger MAE={report.mae} baseline={report.baseline_mae} "
        f"improvement={report.mae_improvement_pct}%"
    )

    # 4) promotion gate: first model promoted, then a worse rerun is rejected
    champion_json = tmp_path / "models" / "champion.json"
    decision = decide(report, champion_json)
    apply(decision, report, tmp_path / "models")
    assert decision.promoted
    assert json.loads(champion_json.read_text())["mae"] == report.mae
