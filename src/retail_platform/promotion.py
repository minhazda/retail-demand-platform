"""Champion/challenger gate: a new model is promoted only if it wins.

The champion's metrics live in ``models/champion.json`` (committed, so the
bar survives across runs). A challenger must beat the champion's holdout MAE
by ``MIN_IMPROVEMENT_PCT`` — retraining alone is never a reason to ship.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .training import TrainReport

MIN_IMPROVEMENT_PCT = 1.0


@dataclass(frozen=True)
class Decision:
    promoted: bool
    reason: str
    challenger_mae: float
    champion_mae: float | None


def decide(challenger: TrainReport, champion_path: Path) -> Decision:
    if not champion_path.exists():
        return Decision(
            True, "no champion on record: first model is promoted", challenger.mae, None
        )

    champion_mae = float(json.loads(champion_path.read_text())["mae"])
    improvement = 100 * (1 - challenger.mae / champion_mae) if champion_mae else 0.0
    if improvement >= MIN_IMPROVEMENT_PCT:
        return Decision(
            True,
            f"challenger MAE {challenger.mae} beats champion {champion_mae} "
            f"by {improvement:.2f}% (>= {MIN_IMPROVEMENT_PCT}%)",
            challenger.mae,
            champion_mae,
        )
    return Decision(
        False,
        f"challenger MAE {challenger.mae} does not beat champion {champion_mae} "
        f"by {MIN_IMPROVEMENT_PCT}% (improvement: {improvement:.2f}%)",
        challenger.mae,
        champion_mae,
    )


def apply(decision: Decision, challenger: TrainReport, models_dir: Path) -> None:
    """On promotion, the challenger bundle and metrics become the champion."""
    if not decision.promoted:
        return
    champion = models_dir / "champion.joblib"
    challenger_bundle = models_dir / "challenger.joblib"
    if challenger_bundle.exists():
        champion.write_bytes(challenger_bundle.read_bytes())
    (models_dir / "champion.json").write_text(json.dumps(asdict(challenger), indent=2))
