"""Dagster entrypoint: the asset graph plus the weekly retrain schedule."""

from __future__ import annotations

from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    define_asset_job,
)

from .assets import challenger_model, dbt_marts, promotion_gate, raw_transactions

retrain_job = define_asset_job("weekly_retrain", selection=AssetSelection.all())

weekly_schedule = ScheduleDefinition(
    job=retrain_job,
    cron_schedule="0 3 * * 1",  # Mondays 03:00 UTC
)

defs = Definitions(
    assets=[raw_transactions, dbt_marts, challenger_model, promotion_gate],
    jobs=[retrain_job],
    schedules=[weekly_schedule],
)
