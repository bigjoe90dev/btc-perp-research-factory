"""
EURUSD MA Cross strategy (bar-based).

Uses M1 bars, fast/slow SMA cross, ATR-based SL/TP, and basic filters.
Designed for deep paper simulator + backtest compatibility.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from collections import deque
import os
from typing import Optional


@dataclass
class Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float
    ticks: int


class CandleAggregator:
    def __init__(self, minutes: int):
        self.seconds = int(minutes * 60)
        self.current_start: Optional[datetime] = None
        self.current_bar: Optional[Bar] = None

    def update(self, ts: datetime, price: float) -> Optional[Bar]:
        bucket = int(ts.timestamp()) // self.seconds * self.seconds
        start = datetime.fromtimestamp(bucket, tz=timezone.utc)

        if self.current_start is None:
            self.current_start = start
            self.current_bar = Bar(start=start, open=price, high=price, low=price, close=price, ticks=1)
            return None

        if start != self.current_start:
            finished = self.current_bar
            self.current_start = start
            self.current_bar = Bar(start=start, open=price, high=price, low=price, close=price, ticks=1)
            return finished

        bar = self.current_bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        bar.ticks += 1
        return None


@dataclass
class MACrossConfig:
    fast_n: int = 20
    slow_n: int = 50
    bar_minutes: int = 1
    atr_period: int = 14
    atr_min: float = 0.0
    atr_max: float = 999.0
    rr: float = 2.0
    sl_atr_mult: float = 2.0

    max_spread: float = 0.0003  # EURUSD: 3 pips = 0.0003
    risk_per_trade: float = 0.005
    max_trades_per_day: int = 4
    max_daily_loss_pct: float = 0.02

    @staticmethod
    def from_env() -> "MACrossConfig":
        def _get(name, default):
            v = os.getenv(name)
            return v if v not in (None, "") else default

        return MACrossConfig(
            fast_n=int(_get("MA_FAST", 20)),
            slow_n=int(_get("MA_SLOW", 50)),
            bar_minutes=int(_get("MA_BAR_MINUTES", 1)),
            atr_period=int(_get("MA_ATR_PERIOD", 14)),
            atr_min=float(_get("MA_ATR_MIN", 0.0)),
            atr_max=float(_get("MA_ATR_MAX", 999.0)),
            rr=float(_get("MA_RR", 2.0)),
            sl_atr_mult=float(_get("MA_SL_ATR_MULT", 2.0)),
            max_spread=float(_get("MA_MAX_SPREAD", 0.0003)),
            risk_per_trade=float(_get("MA_RISK_PER_TRADE", 0.005)),
            max_trades_per_day=int(_get("MA_MAX_TRADES_PER_DAY", 4)),
            max_daily_loss_pct=float(_get("MA_MAX_DAILY_LOSS_PCT", 0.02)),
        )


@dataclass
class EntrySignal:
    side: str
    sl_price: float
    tp_price: float
    reason: str


class MACrossStrategy:
    def __init__(self, cfg: MACrossConfig):
        self.cfg = cfg
        self.agg = CandleAggregator(cfg.bar_minutes)
        self.bars: deque = deque(maxlen=5000)
        self.last_fast: Optional[float] = None
        self.last_slow: Optional[float] = None

    def _atr(self) -> Optional[float]:
        if len(self.bars) < self.cfg.atr_period + 1:
            return None
        bars = list(self.bars)
        prev_close = bars[-self.cfg.atr_period - 1].close
        trs = []
        for b in bars[-self.cfg.atr_period:]:
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
            prev_close = b.close
        return sum(trs) / len(trs)

    def _sma(self, n: int) -> Optional[float]:
        if len(self.bars) < n:
            return None
        closes = [b.close for b in list(self.bars)[-n:]]
        return sum(closes) / len(closes)

    def on_tick(self, ts: datetime, bid: float, ask: float,
                spread_ok: bool = True) -> Optional[EntrySignal]:
        if not spread_ok:
            return None

        mid = (bid + ask) / 2.0
        closed = self.agg.update(ts, mid)
        if not closed:
            return None

        self.bars.append(closed)

        fast = self._sma(self.cfg.fast_n)
        slow = self._sma(self.cfg.slow_n)
        if fast is None or slow is None:
            return None

        atr = self._atr()
        if atr is None or not (self.cfg.atr_min <= atr <= self.cfg.atr_max):
            self.last_fast = fast
            self.last_slow = slow
            return None

        # Detect cross
        signal = None
        if self.last_fast is not None and self.last_slow is not None:
            crossed_up = self.last_fast <= self.last_slow and fast > slow
            crossed_down = self.last_fast >= self.last_slow and fast < slow
            if crossed_up:
                sl = mid - (self.cfg.sl_atr_mult * atr)
                tp = mid + (self.cfg.rr * (mid - sl))
                signal = EntrySignal("BUY", sl, tp, "ma_cross_up")
            elif crossed_down:
                sl = mid + (self.cfg.sl_atr_mult * atr)
                tp = mid - (self.cfg.rr * (sl - mid))
                signal = EntrySignal("SELL", sl, tp, "ma_cross_down")

        self.last_fast = fast
        self.last_slow = slow
        return signal

    def should_time_exit(self, _ts: datetime) -> bool:
        return False
