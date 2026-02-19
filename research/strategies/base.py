from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    target_side: int
    reason: str


class StrategyContext:
    """
    Strict no-lookahead context: exposes data only up to current bar index.
    """

    def __init__(self, frame: pd.DataFrame, idx: int) -> None:
        if idx < 0 or idx >= len(frame):
            raise IndexError("context index out of bounds")
        self._frame = frame
        self._idx = idx

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def ts_utc(self) -> pd.Timestamp:
        return pd.Timestamp(self._frame.iloc[self._idx]["ts_utc"])

    def current(self, field: str) -> float:
        return float(self._frame.iloc[self._idx][field])

    def history(self, field: str, length: int | None = None) -> np.ndarray:
        if field not in self._frame.columns:
            raise KeyError(f"Unknown field: {field}")
        if length is None:
            start = 0
        else:
            start = max(0, self._idx - int(length) + 1)
        arr = self._frame.iloc[start : self._idx + 1][field].to_numpy()
        return arr

    def future(self, *_: Any, **__: Any) -> np.ndarray:
        raise RuntimeError("Future access is forbidden by no-lookahead policy")


class Strategy:
    family: str = "base"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def warmup_bars(self) -> int:
        return 1

    def on_bar(self, ctx: StrategyContext, current_side: int) -> StrategySignal:
        raise NotImplementedError
