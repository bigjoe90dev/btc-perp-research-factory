from __future__ import annotations

import hashlib
import json
from typing import Any


def strategy_id(
    family: str,
    timeframe: str,
    params: dict[str, Any],
    rules_version: str,
    dataset_key: str,
) -> str:
    payload = {
        "family": family,
        "timeframe": timeframe,
        "params": params,
        "rules_version": rules_version,
        "dataset_key": dataset_key,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
