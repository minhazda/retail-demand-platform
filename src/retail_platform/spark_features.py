"""PySpark implementation of the supervised-learning frame.

At this dataset's size (~1M raw transactions) pandas builds the frame in
seconds — Spark buys nothing here. This module exists for the pipeline shape
where the input *doesn't* fit one machine: the same leakage-safe feature
logic expressed in Spark's distributed window API. A parity test pins both
implementations to identical output (tests/test_spark_features.py), so the
pandas module stays the single source of truth for the feature definitions.

Semantics mirrored from :mod:`retail_platform.features`:

- lags and rolling means are positional (rows within a series), not calendar
  — a ``rowsBetween`` window, not ``rangeBetween``;
- ``rolling_mean_w`` averages the *previous* ``w`` rows (pandas
  ``shift(1).rolling(w, min_periods=1)``) → ``rowsBetween(-w, -1)``;
- ``day_of_week`` follows pandas (Monday=0); Spark's ``dayofweek`` is
  Sunday=1, hence the ``+5 % 7`` shift;
- ``series_id`` reproduces pandas category codes: the rank of the stock code
  among sorted distinct codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

from retail_platform.features import LAGS, ROLLING, TARGET

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame


def build_frame_spark(daily: DataFrame) -> DataFrame:
    """``fct_daily_sales`` rows -> model frame; distributed twin of ``build_frame``."""
    series = Window.partitionBy("stock_code").orderBy("invoice_date")

    frame = daily.withColumn("invoice_date", F.to_date("invoice_date"))
    for lag in LAGS:
        frame = frame.withColumn(f"lag_{lag}", F.lag(TARGET, lag).over(series))
    for window in ROLLING:
        trailing = series.rowsBetween(-window, -1)
        frame = frame.withColumn(f"rolling_mean_{window}", F.avg(TARGET).over(trailing))

    frame = frame.withColumn("day_of_week", (F.dayofweek("invoice_date") + F.lit(5)) % 7)
    frame = frame.withColumn("month", F.month("invoice_date"))

    # Rank of the code among sorted distinct codes == pandas category codes.
    # The un-partitioned window is deliberate and cheap: it runs over the
    # distinct codes only (one row per series), not the full frame.
    codes = (
        daily.select("stock_code")
        .distinct()
        .withColumn("series_id", F.dense_rank().over(Window.orderBy("stock_code")) - F.lit(1))
    )
    frame = frame.join(codes, on="stock_code", how="left")

    return frame.where(F.col(f"lag_{max(LAGS)}").isNotNull())
