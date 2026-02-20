from __future__ import annotations

import json
from pathlib import Path

from research.cli import _load_candidates_from_file


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
