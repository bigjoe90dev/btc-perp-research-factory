from __future__ import annotations

import numpy as np
import pandas as pd


def align_funding_to_bars(bar_ts: pd.Series, funding_df: pd.DataFrame) -> np.ndarray:
    """
    Map discrete funding events to bar indices using first bar whose ts >= funding timestamp.
    Returns per-bar summed funding rates.
    """
    if bar_ts.empty:
        return np.zeros(0, dtype=float)

    bar_ns = pd.to_datetime(bar_ts, utc=True).astype("int64").to_numpy()
    fund_ns = pd.to_datetime(funding_df["ts_utc"], utc=True).astype("int64").to_numpy()
    rates = pd.to_numeric(funding_df["funding_rate_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = np.zeros(len(bar_ns), dtype=float)
    idxs = np.searchsorted(bar_ns, fund_ns, side="left")
    for idx, rate in zip(idxs, rates):
        if 0 <= idx < len(out):
            out[idx] += float(rate)
    return out


def funding_cashflow(position_qty: float, mark_price: float, funding_rate: float) -> float:
    """
    Positive cashflow means account receives funding.
    Convention: positive rate => longs pay, shorts receive.
    """
    notional = position_qty * mark_price
    return -notional * float(funding_rate)
