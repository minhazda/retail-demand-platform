"""Environment-driven paths (RDP_* variables) shared by assets, dbt, and CI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    duckdb_path: Path
    dbt_project_dir: Path
    models_dir: Path
    champion_path: Path


def load() -> Settings:
    duckdb_path = Path(os.environ.get("RDP_DUCKDB_PATH", REPO_ROOT / "data" / "warehouse.duckdb"))
    models_dir = Path(os.environ.get("RDP_MODELS_DIR", REPO_ROOT / "models"))
    return Settings(
        duckdb_path=duckdb_path,
        dbt_project_dir=REPO_ROOT / "dbt_project",
        models_dir=models_dir,
        champion_path=models_dir / "champion.json",
    )
