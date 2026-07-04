"""Dagster asset graph: ingest -> dbt (transform + test) -> train -> gate."""

from __future__ import annotations

import duckdb
import pandas as pd
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

from . import ingest as ingest_mod
from .config import load
from .promotion import apply as apply_promotion
from .promotion import decide
from .training import train_challenger


@asset(group_name="platform")
def raw_transactions(context: AssetExecutionContext) -> MaterializeResult:
    """UCI Online Retail II loaded into DuckDB ``raw.transactions``."""
    settings = load()
    df = ingest_mod.fetch_raw(settings.duckdb_path.parent / "raw_cache.parquet")
    rows = ingest_mod.load_to_duckdb(df, settings.duckdb_path)
    context.log.info("loaded %d raw rows", rows)
    return MaterializeResult(metadata={"rows": rows})


@asset(deps=[raw_transactions], group_name="platform")
def dbt_marts(context: AssetExecutionContext) -> MaterializeResult:
    """Star schema built and *tested* by dbt (``dbt build`` runs models + tests)."""
    import os

    from dbt.cli.main import dbtRunner

    settings = load()
    os.environ["RDP_DUCKDB_PATH"] = str(settings.duckdb_path)
    result = dbtRunner().invoke(
        [
            "build",
            "--project-dir",
            str(settings.dbt_project_dir),
            "--profiles-dir",
            str(settings.dbt_project_dir),
        ]
    )
    if not result.success:
        raise RuntimeError(f"dbt build failed: {result.exception}")
    context.log.info("dbt build succeeded (models + tests)")
    return MaterializeResult(metadata={"dbt": "build passed"})


@asset(deps=[dbt_marts], group_name="platform")
def challenger_model(context: AssetExecutionContext) -> MaterializeResult:
    """LightGBM challenger trained on the mart, scored against seasonal-naive."""
    settings = load()
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as conn:
        daily: pd.DataFrame = conn.execute("select * from fct_daily_sales").df()
    report = train_challenger(daily, settings.models_dir)
    context.log.info("challenger: %s", report)
    return MaterializeResult(
        metadata={
            "mae": report.mae,
            "baseline_mae": report.baseline_mae,
            "improvement_pct": report.mae_improvement_pct,
            "n_series": report.n_series,
        }
    )


@asset(deps=[challenger_model], group_name="platform")
def promotion_gate(context: AssetExecutionContext) -> MaterializeResult:
    """Champion/challenger decision; promotes only on a real improvement."""
    import json

    from .training import TrainReport

    settings = load()
    challenger = TrainReport(**json.loads((settings.models_dir / "challenger.json").read_text()))
    decision = decide(challenger, settings.champion_path)
    apply_promotion(decision, challenger, settings.models_dir)
    context.log.info("promotion: %s", decision.reason)
    return MaterializeResult(
        metadata={
            "promoted": decision.promoted,
            "reason": MetadataValue.text(decision.reason),
        }
    )
