"""
GLSB strategy (Gold Liquidity Sweep Breakout) — shared logic.
Designed for XAUUSD but works for any symbol with correct pip config.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque
import os
from typing import Optional, Dict


@dataclass
class Bar:
    start: datetime
    open: float
    high: float
    low: float
    close: float


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
            self.current_bar = Bar(start=start, open=price, high=price, low=price, close=price)
            return None

        if start != self.current_start:
            finished = self.current_bar
            self.current_start = start
            self.current_bar = Bar(start=start, open=price, high=price, low=price, close=price)
            return finished

        bar = self.current_bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        return None


@dataclass
class SessionConfig:
    name: str
    range_start: str
    range_end: str
    trade_start: str
    trade_end: str
    exit_time: str


@dataclass
class SessionState:
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_mid: Optional[float] = None
    range_ready: bool = False
    range_valid: bool = False
    skip_day: bool = False
    swept_high: bool = False
    swept_low: bool = False
    last_breakout_side: Optional[str] = None
    last_breakout_bar_index: Optional[int] = None
    reversion_used: bool = False


@dataclass
class GLSBConfig:
    tz_offset_hours: int = 0
    atr_period: int = 14
    atr_min_usd: float = 0.50
    atr_max_usd: float = 5.00
    ema_period: int = 50
    buffer_usd: float = 0.30
    sweep_buffer_usd: float = 0.10
    min_range_usd: float = 2.50
    max_range_usd: float = 12.00
    max_spread_usd: float = 0.30
    risk_per_trade: float = 0.005
    max_trades_per_day: int = 2
    max_daily_loss_pct: float = 0.02
    rr: float = 2.0
    breakeven_rr: float = 1.0
    trail_pct_range: float = 0.5
    sessions: Dict[str, SessionConfig] = None

    @staticmethod
    def from_env() -> "GLSBConfig":
        def _get(name, default):
            v = os.getenv(name)
            return v if v not in (None, "") else default

        # Default session windows (UTC)
        defaults = {
            "ASIA": SessionConfig("ASIA", "00:00", "02:55", "03:00", "06:00", "06:30"),
            "LDN": SessionConfig("LDN", "06:00", "07:55", "08:00", "11:00", "11:30"),
            "NY": SessionConfig("NY", "12:00", "13:55", "14:00", "17:00", "17:30"),
        }

        raw_sessions = str(_get("GLSB_SESSIONS", "ASIA,LDN,NY"))
        session_keys = [s.strip().upper() for s in raw_sessions.split(",") if s.strip()]
        sessions = {}
        for key in session_keys:
            base = defaults.get(key, SessionConfig(key, "06:00", "07:55", "08:00", "11:00", "11:30"))
            prefix = f"GLSB_{key}_"
            sessions[key] = SessionConfig(
                name=key,
                range_start=str(_get(prefix + "RANGE_START", base.range_start)),
                range_end=str(_get(prefix + "RANGE_END", base.range_end)),
                trade_start=str(_get(prefix + "TRADE_START", base.trade_start)),
                trade_end=str(_get(prefix + "TRADE_END", base.trade_end)),
                exit_time=str(_get(prefix + "EXIT_TIME", base.exit_time)),
            )

        return GLSBConfig(
            tz_offset_hours=int(_get("GLSB_TZ_OFFSET_HOURS", 0)),
            atr_period=int(_get("GLSB_ATR_PERIOD", 14)),
            atr_min_usd=float(_get("GLSB_ATR_MIN_USD", 0.50)),
            atr_max_usd=float(_get("GLSB_ATR_MAX_USD", 5.00)),
            ema_period=int(_get("GLSB_EMA_PERIOD", 50)),
            buffer_usd=float(_get("GLSB_BUFFER_USD", 0.30)),
            sweep_buffer_usd=float(_get("GLSB_SWEEP_BUFFER_USD", 0.10)),
            min_range_usd=float(_get("GLSB_MIN_RANGE_USD", 2.50)),
            max_range_usd=float(_get("GLSB_MAX_RANGE_USD", 12.00)),
            max_spread_usd=float(_get("GLSB_MAX_SPREAD_USD", 0.30)),
            risk_per_trade=float(_get("GLSB_RISK_PER_TRADE", 0.005)),
            max_trades_per_day=int(_get("GLSB_MAX_TRADES_PER_DAY", 2)),
            max_daily_loss_pct=float(_get("GLSB_MAX_DAILY_LOSS_PCT", 0.02)),
            rr=float(_get("GLSB_RR", 2.0)),
            breakeven_rr=float(_get("GLSB_BREAKEVEN_RR", 1.0)),
            trail_pct_range=float(_get("GLSB_TRAIL_PCT_RANGE", 0.5)),
            sessions=sessions,
        )


@dataclass
class EntrySignal:
    side: str  # "BUY" or "SELL"
    sl_price: float
    tp_price: float
    reason: str
    session: str


class GLSBStrategy:
    def __init__(self, cfg: GLSBConfig):
        self.cfg = cfg
        self.bars_5m: deque = deque(maxlen=500)
        self.bars_1h: deque = deque(maxlen=500)
        self.ema_1h: Optional[float] = None
        self.agg_5m = CandleAggregator(5)
        self.agg_1h = CandleAggregator(60)

        self.session_state: Dict[str, SessionState] = {
            name: SessionState() for name in (cfg.sessions or {}).keys()
        }

        self.bar_index: int = 0
        self.session_day: Optional[str] = None

    def reset_for_day(self, ts: datetime):
        for name in self.session_state.keys():
            self.session_state[name] = SessionState()
        self.bar_index = 0
        self.session_day = self._session_date_str(ts)

    def _session_date_str(self, ts: datetime) -> str:
        return (ts + timedelta(hours=self.cfg.tz_offset_hours)).date().isoformat()

    def _session_minutes(self, ts: datetime) -> int:
        st = ts + timedelta(hours=self.cfg.tz_offset_hours)
        return st.hour * 60 + st.minute

    def _minutes_from_str(self, s: str) -> int:
        h, m = s.split(":")
        return int(h) * 60 + int(m)

    def _in_window(self, ts: datetime, start: str, end: str) -> bool:
        m = self._session_minutes(ts)
        return self._minutes_from_str(start) <= m < self._minutes_from_str(end)

    def _after_time(self, ts: datetime, t: str) -> bool:
        return self._session_minutes(ts) >= self._minutes_from_str(t)

    def _update_ema(self, close: float):
        if self.ema_1h is None:
            if len(self.bars_1h) >= self.cfg.ema_period:
                vals = [b.close for b in list(self.bars_1h)[-self.cfg.ema_period:]]
                self.ema_1h = sum(vals) / len(vals)
        else:
            k = 2.0 / (self.cfg.ema_period + 1)
            self.ema_1h = (close * k) + (self.ema_1h * (1 - k))

    def _atr(self) -> Optional[float]:
        if len(self.bars_5m) < self.cfg.atr_period + 1:
            return None
        bars = list(self.bars_5m)
        prev_close = bars[-self.cfg.atr_period - 1].close
        trs = []
        for b in bars[-self.cfg.atr_period:]:
            tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
            trs.append(tr)
            prev_close = b.close
        return sum(trs) / len(trs)

    def _update_range(self, state: SessionState, bar: Bar):
        state.range_high = bar.high if state.range_high is None else max(state.range_high, bar.high)
        state.range_low = bar.low if state.range_low is None else min(state.range_low, bar.low)

    def _finalize_range_if_ready(self, state: SessionState, sess: SessionConfig, bar_end: datetime):
        if state.range_ready or state.skip_day:
            return
        if self._after_time(bar_end, sess.trade_start):
            if state.range_high is None or state.range_low is None:
                state.skip_day = True
                return
            height = state.range_high - state.range_low
            state.range_mid = (state.range_high + state.range_low) / 2.0
            state.range_ready = True
            state.range_valid = self.cfg.min_range_usd <= height <= self.cfg.max_range_usd
            if not state.range_valid:
                state.skip_day = True

    def _update_sweeps(self, state: SessionState, bar: Bar):
        if not state.range_ready:
            return
        if bar.low <= (state.range_low - self.cfg.sweep_buffer_usd):
            state.swept_low = True
        if bar.high >= (state.range_high + self.cfg.sweep_buffer_usd):
            state.swept_high = True

    def _bias(self) -> Optional[str]:
        if self.ema_1h is None or not self.bars_1h:
            return None
        last_close = self.bars_1h[-1].close
        if last_close > self.ema_1h:
            return "LONG"
        if last_close < self.ema_1h:
            return "SHORT"
        return None

    def _evaluate_session(self, state: SessionState, sess: SessionConfig,
                          bar: Bar, atr: float, bias: str) -> Optional[EntrySignal]:
        # Breakout entries
        if bias == "LONG" and state.swept_low:
            if bar.close > (state.range_high + self.cfg.buffer_usd):
                sl_atr = bar.close - (1.5 * atr)
                sl_range = state.range_mid
                sl = max(sl_atr, sl_range)
                tp = bar.close + (self.cfg.rr * (bar.close - sl))
                state.last_breakout_side = "LONG"
                state.last_breakout_bar_index = self.bar_index
                return EntrySignal("BUY", sl, tp, "breakout_long", sess.name)

        if bias == "SHORT" and state.swept_high:
            if bar.close < (state.range_low - self.cfg.buffer_usd):
                sl_atr = bar.close + (1.5 * atr)
                sl_range = state.range_mid
                sl = min(sl_atr, sl_range)
                tp = bar.close - (self.cfg.rr * (sl - bar.close))
                state.last_breakout_side = "SHORT"
                state.last_breakout_bar_index = self.bar_index
                return EntrySignal("SELL", sl, tp, "breakout_short", sess.name)

        # Reversion logic (fakeout within 3 bars)
        if state.last_breakout_side and not state.reversion_used:
            if state.last_breakout_bar_index is not None:
                if (self.bar_index - state.last_breakout_bar_index) <= 3:
                    if state.range_low < bar.close < state.range_high:
                        state.reversion_used = True
                        if state.last_breakout_side == "LONG":
                            sl = state.range_high + self.cfg.buffer_usd
                            tp = state.range_low
                            return EntrySignal("SELL", sl, tp, "reversion_short", sess.name)
                        if state.last_breakout_side == "SHORT":
                            sl = state.range_low - self.cfg.buffer_usd
                            tp = state.range_high
                            return EntrySignal("BUY", sl, tp, "reversion_long", sess.name)

        return None

    def on_tick(self, ts: datetime, bid: float, ask: float,
                paused: bool = False, spread_ok: bool = True,
                has_position: bool = False) -> Optional[EntrySignal]:
        # Daily reset
        day = self._session_date_str(ts)
        if self.session_day != day:
            self.reset_for_day(ts)

        mid = (bid + ask) / 2.0
        closed_5m = self.agg_5m.update(ts, mid)
        closed_1h = self.agg_1h.update(ts, mid)

        if closed_1h:
            self.bars_1h.append(closed_1h)
            self._update_ema(closed_1h.close)

        if not closed_5m:
            return None

        self.bar_index += 1
        self.bars_5m.append(closed_5m)

        # Update ranges and sweeps per session
        for name, sess in self.cfg.sessions.items():
            state = self.session_state[name]
            if self._in_window(closed_5m.start, sess.range_start, sess.range_end):
                self._update_range(state, closed_5m)
            bar_end = closed_5m.start + timedelta(minutes=5)
            self._finalize_range_if_ready(state, sess, bar_end)
            self._update_sweeps(state, closed_5m)

        if paused or not spread_ok or has_position:
            return None

        atr = self._atr()
        if atr is None or not (self.cfg.atr_min_usd <= atr <= self.cfg.atr_max_usd):
            return None

        bias = self._bias()
        if bias is None:
            return None

        # Evaluate active sessions (trade window)
        bar_end = closed_5m.start + timedelta(minutes=5)
        for name, sess in self.cfg.sessions.items():
            state = self.session_state[name]
            if state.skip_day or not state.range_ready:
                continue
            if not self._in_window(bar_end, sess.trade_start, sess.trade_end):
                continue
            signal = self._evaluate_session(state, sess, closed_5m, atr, bias)
            if signal:
                return signal

        return None

    def should_time_exit(self, ts: datetime, session: Optional[str] = None) -> bool:
        if session and session in self.cfg.sessions:
            return self._after_time(ts, self.cfg.sessions[session].exit_time)

        # fallback: after latest exit time
        latest = None
        for sess in self.cfg.sessions.values():
            if latest is None or self._minutes_from_str(sess.exit_time) > self._minutes_from_str(latest):
                latest = sess.exit_time
        if latest:
            return self._after_time(ts, latest)
        return False
