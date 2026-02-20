from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.data.loader import load_dataset_from_manifest


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_loader_rejects_dataset_symbol_or_market_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yml"
    data_cfg_path = tmp_path / "data.yml"

    manifest = {
        "datasets": [
            {
                "dataset_id": "BAD_SCOPE",
                "symbol": "ETH",
                "market": "spot",
                "canonical_file": str(tmp_path / "missing.parquet"),
                "funding_file": str(tmp_path / "missing_funding.parquet"),
            }
        ]
    }
    _write_yaml(manifest_path, manifest)

    data_cfg = {
        "manifest_path": str(manifest_path),
        "primary_dataset_key": "BAD_SCOPE",
        "required_symbol": "BTC",
        "required_market": "perp",
        "loader": {"enforce_funding_coverage": True},
    }
    _write_yaml(data_cfg_path, data_cfg)

    with pytest.raises(RuntimeError, match="Dataset symbol mismatch"):
        load_dataset_from_manifest(data_config_path=data_cfg_path, timeframe="1m")


def test_loader_fails_closed_on_funding_coverage_gap(tmp_path: Path) -> None:
    candles_path = tmp_path / "candles.parquet"
    funding_path = tmp_path / "funding.parquet"
    manifest_path = tmp_path / "manifest.yml"
    data_cfg_path = tmp_path / "data.yml"

    ts = pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="min")
    candles = pd.DataFrame(
        {
            "ts_utc": ts,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    candles.to_parquet(candles_path, index=False)

    # Funding starts too late and does not span candle window.
    funding = pd.DataFrame(
        {
            "ts_utc": [pd.Timestamp("2026-01-01T00:03:00Z")],
            "funding_rate_raw": [0.0001],
        }
    )
    funding.to_parquet(funding_path, index=False)

    manifest = {
        "datasets": [
            {
                "dataset_id": "BTC_TEST",
                "symbol": "BTC",
                "market": "perp",
                "canonical_file": str(candles_path),
                "funding_file": str(funding_path),
            }
        ]
    }
    _write_yaml(manifest_path, manifest)

    data_cfg = {
        "manifest_path": str(manifest_path),
        "primary_dataset_key": "BTC_TEST",
        "required_symbol": "BTC",
        "required_market": "perp",
        "loader": {
            "enforce_funding_coverage": True,
            "funding_end_tolerance_hours": 0,
        },
    }
    _write_yaml(data_cfg_path, data_cfg)

    with pytest.raises(RuntimeError, match="Funding coverage does not fully span candle coverage"):
        load_dataset_from_manifest(data_config_path=data_cfg_path, timeframe="1m")
