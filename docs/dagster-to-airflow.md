# Dagster → Airflow: how this pipeline maps

This platform is orchestrated with Dagster assets. Many teams run Airflow.
The point of this note: the design decisions in this repo — data tests as a
pipeline stage, a promotion gate between training and deployment, weekly
scheduled retrains — are orchestrator-independent. Here is the concrete
translation, asset by asset.

## Concept mapping

| This repo (Dagster) | Airflow equivalent | Notes |
|---|---|---|
| `@asset raw_transactions` | `PythonOperator` task (or `@task`) | Ingest step; same function body |
| `@asset dbt_marts` (dagster-dbt) | `BashOperator` running `dbt build`, or Cosmos | `dbt build` failing fails the task either way — the "data tests as a stage" decision survives |
| `@asset challenger_model` | `@task` downstream of dbt | Identical training code |
| `@asset promotion_gate` | `BranchPythonOperator` | Promote vs. stop is a branch, not an edge |
| Asset dependencies (function args) | `task_a >> task_b` edges | Dagster infers the graph from data deps; Airflow declares it |
| Asset materialization + metadata | XCom + task logs | Dagster persists per-asset metadata natively; Airflow needs XCom or an external store |
| `ScheduleDefinition` (weekly) | `schedule="0 10 * * 1"` on the DAG | Same cron, different spelling |
| Asset checks / freshness | SLAs + `Dataset` aware scheduling | Airflow Datasets (2.4+) are the closest analog to asset-based thinking |
| `dagster dev` UI (asset lineage) | Airflow UI (task view) + Datasets view | Lineage is first-class in Dagster; assembled in Airflow |

## The one real difference

Dagster models **what exists** (assets: tables, models, files) and derives
the run order; Airflow models **what runs** (tasks) and you wire the order.
Everything else in this repo — the champion/challenger gate, versioned
champion metrics in git, dbt tests failing the pipeline before training sees
bad rows, mirror-first ingestion — transfers as-is: those are pipeline
*design* decisions, expressed in whichever orchestrator's API a team uses.

## What an Airflow port of this repo would look like

One DAG (`retail_demand_weekly`), five tasks:

```
ingest_raw >> dbt_build >> train_challenger >> promotion_gate >> [commit_champion, stop]
```

with `dbt_build` as a `BashOperator` (`dbt build --profiles-dir .`),
`promotion_gate` as a `BranchPythonOperator` calling
`retail_platform.promotion.decide()` unchanged, and the weekly cron moved
from `.github/workflows/retrain.yml` onto the DAG schedule.
