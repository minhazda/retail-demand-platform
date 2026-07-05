"""Parity: the PySpark frame must equal the pandas frame, value for value.

The pandas implementation is the source of truth for feature semantics;
this suite proves the distributed implementation computes the same thing —
including the awkward corners (series shorter than the longest lag, gaps in
the calendar, day-of-week convention, category-code ordering).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from retail_platform.features import TARGET, build_frame, feature_columns

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    # Workers must run the same interpreter as the driver — without this,
    # Spark spawns whatever `python` is on PATH (a different version on
    # machines with multiple Pythons) and the workers crash.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    session = (
        SparkSession.builder.master("local[2]")
        .appName("spark-features-parity")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _synthetic_daily(seed: int = 7) -> pd.DataFrame:
    """3 uneven series incl. calendar gaps and one series shorter than max lag."""
    rng = np.random.default_rng(seed)
    rows = []
    specs = [("SKU_B", 60, 0), ("SKU_A", 45, 3), ("SKU_C", 10, 0)]  # name, days, gap every n
    for code, days, gap_every in specs:
        dates = pd.date_range("2011-01-01", periods=days, freq="D")
        if gap_every:
            dates = dates[[i for i in range(days) if i % gap_every != 1]]
        for d in dates:
            rows.append(
                {
                    "stock_code": code,
                    "invoice_date": d.date().isoformat(),
                    TARGET: float(rng.integers(0, 40)),
                }
            )
    return pd.DataFrame(rows)


def test_spark_frame_matches_pandas_frame(spark) -> None:
    from retail_platform.spark_features import build_frame_spark

    daily = _synthetic_daily()

    expected = build_frame(daily).sort_values(["stock_code", "invoice_date"]).reset_index(drop=True)
    actual = (
        build_frame_spark(spark.createDataFrame(daily))
        .toPandas()
        .sort_values(["stock_code", "invoice_date"])
        .reset_index(drop=True)
    )

    # Same rows survive the max-lag cutoff (the 10-day series drops entirely).
    assert len(actual) == len(expected)
    assert (actual["stock_code"] == expected["stock_code"]).all()
    assert (
        pd.to_datetime(actual["invoice_date"]) == pd.to_datetime(expected["invoice_date"])
    ).all()

    for col in feature_columns(expected) + [TARGET]:
        np.testing.assert_allclose(
            actual[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            atol=1e-9,
            err_msg=f"column {col} diverged between Spark and pandas",
        )


def test_short_series_is_dropped_by_max_lag(spark) -> None:
    from retail_platform.spark_features import build_frame_spark

    daily = _synthetic_daily()
    actual = build_frame_spark(spark.createDataFrame(daily)).toPandas()
    assert "SKU_C" not in set(actual["stock_code"])  # 10 rows < lag_28
