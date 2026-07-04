"""Champion/challenger gate logic."""

from __future__ import annotations

import json
import time
from pathlib import Path

from retail_platform.promotion import MIN_IMPROVEMENT_PCT, apply, decide
from retail_platform.training import TrainReport


def _report(mae: float) -> TrainReport:
    return TrainReport(
        mae=mae,
        rmse=mae * 1.4,
        baseline_mae=10.0,
        mae_improvement_pct=0.0,
        n_train=1000,
        n_test=100,
        n_series=8,
        trained_at=time.time(),
    )


def test_first_model_is_always_promoted(tmp_path: Path) -> None:
    decision = decide(_report(5.0), tmp_path / "champion.json")
    assert decision.promoted and decision.champion_mae is None


def test_better_challenger_promoted(tmp_path: Path) -> None:
    champion = tmp_path / "champion.json"
    champion.write_text(json.dumps({"mae": 10.0}))
    assert decide(_report(8.0), champion).promoted


def test_marginal_challenger_rejected(tmp_path: Path) -> None:
    champion = tmp_path / "champion.json"
    champion.write_text(json.dumps({"mae": 10.0}))
    marginal = 10.0 * (1 - (MIN_IMPROVEMENT_PCT - 0.5) / 100)
    decision = decide(_report(marginal), champion)
    assert not decision.promoted


def test_apply_updates_champion_only_on_promotion(tmp_path: Path) -> None:
    report = _report(5.0)
    (tmp_path / "challenger.joblib").write_bytes(b"bundle")
    decision = decide(report, tmp_path / "champion.json")
    apply(decision, report, tmp_path)
    assert json.loads((tmp_path / "champion.json").read_text())["mae"] == 5.0
    assert (tmp_path / "champion.joblib").read_bytes() == b"bundle"

    worse = _report(9.0)
    rejected = decide(worse, tmp_path / "champion.json")
    apply(rejected, worse, tmp_path)
    assert json.loads((tmp_path / "champion.json").read_text())["mae"] == 5.0  # unchanged
