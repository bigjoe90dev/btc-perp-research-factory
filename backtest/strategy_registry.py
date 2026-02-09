"""
Strategy registry and parameter grids for robustness testing.
"""
import random
from typing import Callable, Dict, List

from backtest.eurusd_strategies import (
    LondonOpenBreakoutStrategy, LondonOpenBreakoutConfig,
    AsianRangeFadeStrategy, AsianRangeFadeConfig,
    MACrossADXStrategy, MACrossADXConfig,
    BollingerBounceStrategy, BollingerBounceConfig,
    RSIDivergenceStrategy, RSIDivergenceConfig,
)


def strategy_factory(name: str) -> Callable[..., object]:
    name = name.lower().strip()
    if name == "london_open_breakout":
        return lambda **p: LondonOpenBreakoutStrategy(LondonOpenBreakoutConfig(**p))
    if name == "asian_range_fade":
        return lambda **p: AsianRangeFadeStrategy(AsianRangeFadeConfig(**p))
    if name == "ma_cross_adx":
        return lambda **p: MACrossADXStrategy(MACrossADXConfig(**p))
    if name == "bollinger_bounce":
        return lambda **p: BollingerBounceStrategy(BollingerBounceConfig(**p))
    if name == "rsi_divergence":
        return lambda **p: RSIDivergenceStrategy(RSIDivergenceConfig(**p))
    return lambda **p: MACrossADXStrategy(MACrossADXConfig(**p))


def param_grid(name: str) -> List[Dict]:
    name = name.lower().strip()
    grid = []
    if name == "london_open_breakout":
        windows = [
            ("06:00", "07:00", "07:00", "10:00"),
            ("07:00", "08:00", "08:00", "11:00"),
            ("07:30", "08:30", "08:30", "11:30"),
        ]
        for buffer_pips in (1.0, 2.0, 3.0):
            for rr in (1.5, 2.0, 2.5):
                for rs, re, ts, te in windows:
                    grid.append({
                        "buffer_pips": buffer_pips,
                        "rr": rr,
                        "range_start": rs,
                        "range_end": re,
                        "trade_start": ts,
                        "trade_end": te,
                    })
    elif name == "asian_range_fade":
        for buffer_pips in (0.5, 1.0, 1.5):
            for lookback_hours in (3, 4, 5):
                for max_spread_pips in (2.0, 3.0):
                    grid.append({
                        "buffer_pips": buffer_pips,
                        "lookback_hours": lookback_hours,
                        "max_spread_pips": max_spread_pips,
                    })
    elif name == "ma_cross_adx":
        for fast in (10, 20, 30):
            for slow in (40, 60, 80):
                for adx_min in (18.0, 22.0, 25.0):
                    if slow > fast:
                        grid.append({"fast": fast, "slow": slow, "adx_min": adx_min})
    elif name == "bollinger_bounce":
        for period in (20, 30):
            for k in (1.8, 2.0, 2.2):
                for band_walk_bars in (2, 3, 4):
                    grid.append({"period": period, "k": k, "band_walk_bars": band_walk_bars})
    elif name == "rsi_divergence":
        for rsi_period in (14, 21):
            for min_rsi_diff in (4.0, 6.0, 8.0):
                for min_price_pips in (4.0, 6.0, 8.0):
                    grid.append({
                        "rsi_period": rsi_period,
                        "min_rsi_diff": min_rsi_diff,
                        "min_price_pips": min_price_pips,
                    })
    return grid or [{}]


def param_sampler(name: str):
    name = name.lower().strip()

    def sampler(rng: random.Random) -> Dict:
        if name == "london_open_breakout":
            windows = [
                ("06:00", "07:00", "07:00", "10:00"),
                ("07:00", "08:00", "08:00", "11:00"),
                ("07:30", "08:30", "08:30", "11:30"),
            ]
            rs, re, ts, te = rng.choice(windows)
            return {
                "buffer_pips": rng.choice([1.0, 2.0, 3.0]),
                "rr": rng.choice([1.5, 2.0, 2.5]),
                "range_start": rs,
                "range_end": re,
                "trade_start": ts,
                "trade_end": te,
            }
        if name == "asian_range_fade":
            return {
                "buffer_pips": rng.choice([0.5, 1.0, 1.5]),
                "lookback_hours": rng.choice([3, 4, 5]),
                "max_spread_pips": rng.choice([2.0, 3.0]),
            }
        if name == "ma_cross_adx":
            fast = rng.choice([10, 20, 30])
            slow = rng.choice([40, 60, 80])
            return {"fast": fast, "slow": slow, "adx_min": rng.choice([18.0, 22.0, 25.0])}
        if name == "bollinger_bounce":
            return {"period": rng.choice([20, 30]), "k": rng.choice([1.8, 2.0, 2.2]), "band_walk_bars": rng.choice([2, 3, 4])}
        if name == "rsi_divergence":
            return {"rsi_period": rng.choice([14, 21]), "min_rsi_diff": rng.choice([4.0, 6.0, 8.0]), "min_price_pips": rng.choice([4.0, 6.0, 8.0])}
        return {}

    return sampler
