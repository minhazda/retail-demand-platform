"""Synthetic fixtures: a mini raw dataset with real weekly seasonality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """~200 days x 8 products of invoice lines, plus known-dirty rows."""
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for product in range(8):
        stock_code = f"1000{product}"
        base = rng.uniform(5, 40)
        for day in range(200):
            date = start + pd.Timedelta(days=day)
            weekly = 1.0 + 0.5 * np.sin(2 * np.pi * date.dayofweek / 7)
            for _ in range(rng.poisson(2) + 1):
                quantity = max(int(rng.poisson(base * weekly / 2)), 1)
                rows.append(
                    {
                        "Invoice": f"{500000 + day}",
                        "StockCode": stock_code,
                        "Description": f"PRODUCT {product}",
                        "Quantity": quantity,
                        "InvoiceDate": date + pd.Timedelta(hours=int(rng.integers(8, 20))),
                        "Price": round(float(rng.uniform(1, 20)), 2),
                        "Customer ID": float(rng.integers(10000, 20000)),
                        "Country": "United Kingdom",
                    }
                )
    dirty = [
        {  # cancellation: must be dropped by staging
            "Invoice": "C999999",
            "StockCode": "10000",
            "Description": "PRODUCT 0",
            "Quantity": -5,
            "InvoiceDate": start,
            "Price": 5.0,
            "Customer ID": 10001.0,
            "Country": "United Kingdom",
        },
        {  # non-product code: must be dropped by staging
            "Invoice": "600000",
            "StockCode": "POST",
            "Description": "POSTAGE",
            "Quantity": 1,
            "InvoiceDate": start,
            "Price": 18.0,
            "Customer ID": 10002.0,
            "Country": "France",
        },
    ]
    return pd.DataFrame(rows + dirty)
