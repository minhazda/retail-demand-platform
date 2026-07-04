"""Supervised-learning frame from the ``fct_daily_sales`` mart.

Per-series lag/rolling features only ever look backwards (shifted before
rolling), so the frame is leakage-safe by construction.
"""

from __future__ import annotations

import pandas as pd

LAGS = (1, 7, 28)
ROLLING = (7, 28)
TARGET = "units_sold"


def build_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """``fct_daily_sales`` rows -> model frame with lags, rolling means, calendar."""
    frame = daily.sort_values(["stock_code", "invoice_date"]).copy()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])

    grouped = frame.groupby("stock_code")[TARGET]
    for lag in LAGS:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in ROLLING:
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )

    frame["day_of_week"] = frame["invoice_date"].dt.dayofweek
    frame["month"] = frame["invoice_date"].dt.month
    frame["series_id"] = frame["stock_code"].astype("category").cat.codes

    return frame.dropna(subset=[f"lag_{max(LAGS)}"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    calendar = ["day_of_week", "month", "series_id"]
    engineered = [c for c in frame.columns if c.startswith(("lag_", "rolling_mean_"))]
    return engineered + calendar
