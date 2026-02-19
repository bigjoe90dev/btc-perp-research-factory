from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    canonical_file: Path
    funding_file: Path | None
    start_ts_utc: str | None
    end_ts_utc: str | None
    funding_start_ts_utc: str | None
    funding_end_ts_utc: str | None


@dataclass
class DataBundle:
    dataset: DatasetInfo
    candles_1m: pd.DataFrame
    candles: pd.DataFrame
    funding: pd.DataFrame
    timeframe: str


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {p}")
    return data


def _find_dataset(manifest: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("Manifest missing datasets list")
    for entry in datasets:
        if isinstance(entry, dict) and str(entry.get("dataset_id")) == dataset_id:
            return entry
    available = [str(x.get("dataset_id")) for x in datasets if isinstance(x, dict)]
    raise ValueError(f"Dataset key not found in manifest: {dataset_id}. Available={available}")


def _load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts_utc_ns" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["ts_utc_ns"], errors="coerce"), unit="ns", utc=True)
    elif "ts_utc" in df.columns:
        ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    else:
        raise ValueError(f"Candle file has no timestamp column: {path}")

    out = df.copy()
    out["ts_utc"] = ts
    req = ["open", "high", "low", "close", "volume"]
    for c in req:
        if c not in out.columns:
            raise ValueError(f"Candle file missing column {c}: {path}")
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["ts_utc", "open", "high", "low", "close", "volume"])
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)

    # Treat source 1m timestamps as bar-open time and store analysis timestamp as bar-close.
    out["ts_utc"] = out["ts_utc"] + pd.Timedelta(minutes=1)
    return out[["ts_utc", "open", "high", "low", "close", "volume"]]


def _load_funding(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts_utc_ns" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["ts_utc_ns"], errors="coerce"), unit="ns", utc=True)
    elif "ts_utc" in df.columns:
        ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    else:
        raise ValueError(f"Funding file has no timestamp column: {path}")

    if "funding_rate_raw" not in df.columns:
        raise ValueError(f"Funding file missing funding_rate_raw: {path}")

    out = df.copy()
    out["ts_utc"] = ts
    out["funding_rate_raw"] = pd.to_numeric(out["funding_rate_raw"], errors="coerce")
    out = out.dropna(subset=["ts_utc", "funding_rate_raw"])
    out = out.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last").reset_index(drop=True)
    return out[["ts_utc", "funding_rate_raw"]]


def _resample_candles(candles_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1m":
        return candles_1m.copy()

    rule_map = {"5m": "5min", "1h": "1h"}
    if timeframe not in rule_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    work = candles_1m.set_index("ts_utc")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = work.resample(rule_map[timeframe], label="right", closed="right").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def load_dataset_from_manifest(
    data_config_path: str | Path,
    timeframe: str,
) -> DataBundle:
    data_cfg = _read_yaml(data_config_path)
    manifest_path = Path(str(data_cfg.get("manifest_path", "research/data_manifest.yml")))
    dataset_key = str(data_cfg.get("primary_dataset_key", "BTC_HYPERLIQUID_PERP_1M"))

    manifest = _read_yaml(manifest_path)
    ds = _find_dataset(manifest, dataset_key)

    canonical_file = Path(str(ds.get("canonical_file")))
    funding_file = Path(str(ds.get("funding_file"))) if ds.get("funding_file") else None

    if not canonical_file.exists():
        raise FileNotFoundError(
            f"Primary candle dataset missing: {canonical_file}. Run downloader before backtests."
        )
    if funding_file is None or not funding_file.exists():
        raise FileNotFoundError(
            f"Funding dataset missing: {funding_file}. Run downloader before backtests."
        )

    candles_1m = _load_candles(canonical_file)
    funding = _load_funding(funding_file)

    if candles_1m.empty:
        raise RuntimeError("Primary candle dataset is empty")
    if funding.empty:
        raise RuntimeError("Funding dataset is empty")

    # Fail-closed for funding coverage on testable candle range.
    c_start = candles_1m["ts_utc"].iloc[0]
    c_end = candles_1m["ts_utc"].iloc[-1]
    f_start = funding["ts_utc"].iloc[0]
    f_end = funding["ts_utc"].iloc[-1]
    enforce_cov = bool(data_cfg.get("loader", {}).get("enforce_funding_coverage", True))
    tol_hours = float(data_cfg.get("loader", {}).get("funding_end_tolerance_hours", 8.0))
    tol = pd.Timedelta(hours=max(tol_hours, 0.0))
    if enforce_cov and (f_start > c_start or (f_end + tol) < c_end):
        raise RuntimeError(
            "Funding coverage does not fully span candle coverage: "
            f"candles=[{c_start},{c_end}] funding=[{f_start},{f_end}] tolerance={tol}"
        )

    candles = _resample_candles(candles_1m, timeframe=timeframe)

    info = DatasetInfo(
        dataset_id=str(ds.get("dataset_id")),
        canonical_file=canonical_file,
        funding_file=funding_file,
        start_ts_utc=ds.get("start_ts_utc"),
        end_ts_utc=ds.get("end_ts_utc"),
        funding_start_ts_utc=ds.get("funding_start_ts_utc"),
        funding_end_ts_utc=ds.get("funding_end_ts_utc"),
    )
    return DataBundle(dataset=info, candles_1m=candles_1m, candles=candles, funding=funding, timeframe=timeframe)


def detect_datasets_from_manifest(data_config_path: str | Path) -> list[dict[str, Any]]:
    data_cfg = _read_yaml(data_config_path)
    manifest_path = Path(str(data_cfg.get("manifest_path", "research/data_manifest.yml")))
    manifest = _read_yaml(manifest_path)

    out: list[dict[str, Any]] = []
    for entry in manifest.get("datasets", []):
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "dataset_id": entry.get("dataset_id"),
                "canonical_file": entry.get("canonical_file"),
                "funding_file": entry.get("funding_file"),
                "start_ts_utc": entry.get("start_ts_utc"),
                "end_ts_utc": entry.get("end_ts_utc"),
                "funding_start_ts_utc": entry.get("funding_start_ts_utc"),
                "funding_end_ts_utc": entry.get("funding_end_ts_utc"),
            }
        )
    return out
