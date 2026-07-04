"""Challenger training: LightGBM vs a seasonal-naive baseline on a time split.

The last ``HOLDOUT_DAYS`` days are held out; the baseline predicts lag-7
(same weekday last week). Metrics are pooled across all series.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import TARGET, build_frame, feature_columns

HOLDOUT_DAYS = 28
SEED = 7


@dataclass(frozen=True)
class TrainReport:
    mae: float
    rmse: float
    baseline_mae: float
    mae_improvement_pct: float
    n_train: int
    n_test: int
    n_series: int
    trained_at: float


def train_challenger(daily: pd.DataFrame, out_dir: Path) -> TrainReport:
    """Train on all but the last month, evaluate against seasonal-naive, persist."""
    frame = build_frame(daily)
    columns = feature_columns(frame)

    cutoff = frame["invoice_date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, test = frame[frame["invoice_date"] <= cutoff], frame[frame["invoice_date"] > cutoff]
    if len(train) < 100 or len(test) < 10:
        raise ValueError(f"not enough data to train/evaluate: {len(train)}/{len(test)} rows")

    model = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[columns], train[TARGET])

    predictions = np.clip(model.predict(test[columns]), 0, None)
    truth = test[TARGET].to_numpy(dtype=float)
    baseline = test["lag_7"].to_numpy(dtype=float)  # same weekday, last week

    mae = float(np.mean(np.abs(truth - predictions)))
    baseline_mae = float(np.mean(np.abs(truth - baseline)))
    report = TrainReport(
        mae=round(mae, 4),
        rmse=round(float(np.sqrt(np.mean((truth - predictions) ** 2))), 4),
        baseline_mae=round(baseline_mae, 4),
        mae_improvement_pct=round(100 * (1 - mae / baseline_mae), 2) if baseline_mae else 0.0,
        n_train=len(train),
        n_test=len(test),
        n_series=int(frame["stock_code"].nunique()),
        trained_at=time.time(),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": columns}, out_dir / "challenger.joblib")
    (out_dir / "challenger.json").write_text(json.dumps(asdict(report), indent=2))
    return report
