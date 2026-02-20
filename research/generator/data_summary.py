from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.data.funding import align_funding_to_bars
from research.data.loader import load_dataset_from_manifest


@dataclass(frozen=True)
class DataSummary:
    timeframe: str
    summary: dict[str, Any]

    def as_json(self) -> str:
        return json.dumps(self.summary, indent=2, sort_keys=True)


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    if len(series) <= lag:
        return 0.0
    val = series.autocorr(lag=lag)
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return 0.0
    return float(val)


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x


def _hurst_exponent(prices: pd.Series) -> float:
    arr = prices.to_numpy(dtype=float)
    if len(arr) < 200:
        return 0.5
    lags = np.array([2, 4, 8, 16, 32], dtype=int)
    tau: list[float] = []
    for lag in lags:
        diff = arr[lag:] - arr[:-lag]
        sd = np.std(diff)
        if sd <= 0:
            continue
        tau.append(sd)
    if len(tau) < 2:
        return 0.5
    x = np.log(lags[: len(tau)])
    y = np.log(np.array(tau))
    slope, _ = np.polyfit(x, y, 1)
    return float(max(min(slope, 1.5), 0.0))


def build_data_summary(data_config_path: str, timeframe: str) -> DataSummary:
    bundle = load_dataset_from_manifest(data_config_path=data_config_path, timeframe=timeframe)
    candles = bundle.candles.copy()
    funding = bundle.funding.copy()

    funding_by_bar = align_funding_to_bars(candles["ts_utc"], funding)
    ret = candles["close"].pct_change().fillna(0.0)
    realized_vol_24 = ret.rolling(24).std().dropna()

    lookback_90 = 90 * 24 if timeframe == "1h" else 90 * 24 * 12 if timeframe == "5m" else 90 * 24 * 60
    recent_funding = funding_by_bar.tail(lookback_90) if lookback_90 > 0 else funding_by_bar

    funding_std = float(funding_by_bar.std(ddof=0)) if len(funding_by_bar) > 1 else 0.0
    funding_mean = float(funding_by_bar.mean()) if len(funding_by_bar) > 0 else 0.0
    if funding_std > 0:
        funding_z = (funding_by_bar - funding_mean) / funding_std
    else:
        funding_z = pd.Series(np.zeros(len(funding_by_bar)), index=funding_by_bar.index)

    volume_q95 = float(candles["volume"].quantile(0.95)) if len(candles) else 0.0

    summary = {
        "dataset_id": bundle.dataset.dataset_id,
        "timeframe": timeframe,
        "date_range": {
            "start": str(pd.Timestamp(candles["ts_utc"].iloc[0])) if len(candles) else None,
            "end": str(pd.Timestamp(candles["ts_utc"].iloc[-1])) if len(candles) else None,
        },
        "num_bars": int(len(candles)),
        "num_funding_points": int(len(funding)),
        "funding_mean": _safe_float(funding_mean),
        "funding_std": _safe_float(funding_std),
        "funding_positive_pct": _safe_float((funding_by_bar > 0).mean() * 100.0),
        "funding_autocorr_8h": _safe_autocorr(funding_by_bar, lag=(8 if timeframe == "1h" else 96 if timeframe == "5m" else 480)),
        "funding_zscore_persistence": _safe_autocorr(funding_z, lag=1),
        "recent_funding_bias": "positive" if _safe_float(recent_funding.mean()) >= 0 else "negative",
        "recent_vs_full_funding_delta": _safe_float(_safe_float(recent_funding.mean()) - _safe_float(funding_by_bar.mean())),
        "avg_daily_range_pct": _safe_float((((candles["high"] - candles["low"]) / candles["close"].replace(0, np.nan)).mean()) * 100.0),
        "vol_clustering": _safe_autocorr(realized_vol_24, lag=1),
        "return_skewness": _safe_float(ret.skew()),
        "return_kurtosis": _safe_float(ret.kurt()),
        "volume_spike_threshold_95": _safe_float(volume_q95),
        "avg_volume": _safe_float(candles["volume"].mean()),
        "hurst_exponent": _safe_float(_hurst_exponent(candles["close"])),
    }
    return DataSummary(timeframe=timeframe, summary=summary)
