from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.cli import _load_candidates_from_file, _validate_candidate_specs_for_run


def test_load_candidates_from_file_dict_format(tmp_path: Path) -> None:
    p = tmp_path / "candidates.json"
    payload = {
        "generation_id": "gen_test",
        "candidates": [
            {
                "strategy_id": "abc123",
                "family": "momentum_breakout",
                "timeframe": "1h",
                "params": {
                    "breakout_lookback": 50,
                    "atr_lookback": 20,
                    "atr_min_pct": 0.001,
                    "time_stop_bars": 48,
                    "trailing_stop_atr": 2.0,
                },
                "rules_version": "v1",
                "dataset_key": "BTC_BITMEX_PERP_1M",
            }
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_candidates_from_file(p)
    assert len(rows) == 1
    assert rows[0].strategy_id == "abc123"


def test_validate_candidate_specs_for_run_rejects_dataset_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "candidates.json"
    payload = {
        "generation_id": "gen_test",
        "candidates": [
            {
                "strategy_id": "abc123",
                "family": "momentum_breakout",
                "timeframe": "1h",
                "params": {
                    "breakout_lookback": 50,
                    "atr_lookback": 20,
                    "atr_min_pct": 0.001,
                    "time_stop_bars": 48,
                    "trailing_stop_atr": 2.0,
                },
                "rules_version": "v1",
                "dataset_key": "WRONG_DATASET",
            }
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_candidates_from_file(p)
    with pytest.raises(RuntimeError, match="dataset_key mismatch"):
        _validate_candidate_specs_for_run(
            candidates=rows,
            dataset_key="BTC_BITMEX_PERP_1M",
            rules_version="v1",
            allowed_timeframes={"1h"},
        )


def test_validate_candidate_specs_for_run_rejects_rules_version_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "candidates.json"
    payload = {
        "generation_id": "gen_test",
        "candidates": [
            {
                "strategy_id": "abc123",
                "family": "momentum_breakout",
                "timeframe": "1h",
                "params": {
                    "breakout_lookback": 50,
                    "atr_lookback": 20,
                    "atr_min_pct": 0.001,
                    "time_stop_bars": 48,
                    "trailing_stop_atr": 2.0,
                },
                "rules_version": "v2",
                "dataset_key": "BTC_BITMEX_PERP_1M",
            }
        ],
    }
    p.write_text(json.dumps(payload), encoding="utf-8")

    rows = _load_candidates_from_file(p)
    with pytest.raises(RuntimeError, match="rules_version mismatch"):
        _validate_candidate_specs_for_run(
            candidates=rows,
            dataset_key="BTC_BITMEX_PERP_1M",
            rules_version="v1",
            allowed_timeframes={"1h"},
        )
