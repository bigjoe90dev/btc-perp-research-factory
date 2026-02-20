from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import Trade


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _annualized_sharpe(ret: pd.Series, bars_per_year: int) -> float:
    if ret.empty:
        return 0.0
    mu = float(ret.mean())
    sd = float(ret.std(ddof=1))
    if sd <= 0:
        return 0.0
    return (mu / sd) * np.sqrt(max(bars_per_year, 1))


def _annualized_sortino(ret: pd.Series, bars_per_year: int) -> float:
    if ret.empty:
        return 0.0
    mu = float(ret.mean())
    downside = ret[ret < 0]
    dd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    if dd <= 0:
        return 0.0
    return (mu / dd) * np.sqrt(max(bars_per_year, 1))


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    bars_per_year: int,
) -> dict[str, Any]:
    if equity_curve.empty:
        return {"ok": False, "reason": "empty_equity_curve"}

    eq = equity_curve.copy()
    eq = eq.sort_values("ts_utc").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change().fillna(0.0)
    starting_equity = float(eq["equity"].iloc[0])

    total_return = _safe_div(float(eq["equity"].iloc[-1] - eq["equity"].iloc[0]), float(eq["equity"].iloc[0]))
    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax.replace(0, np.nan)
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0

    sharpe = _annualized_sharpe(eq["ret"], bars_per_year)
    sortino = _annualized_sortino(eq["ret"], bars_per_year)

    trades_count = len(trades)
    pnl_after = np.array([t.pnl_after_costs for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnl_after[pnl_after > 0]
    losses = pnl_after[pnl_after < 0]

    win_rate = float((pnl_after > 0).mean()) if trades_count > 0 else 0.0
    avg_trade = float(pnl_after.mean()) if trades_count > 0 else 0.0
    profit_factor = _safe_div(float(wins.sum()), abs(float(losses.sum()))) if len(losses) > 0 else float("inf")
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    expectancy = avg_trade
    win_loss_ratio = _safe_div(avg_win, abs(avg_loss))

    exposure = float((eq["gross_notional"] > 0).mean()) if "gross_notional" in eq.columns else 0.0
    turnover = float(eq.get("trade_notional", pd.Series([0.0])).sum())
    turnover_ratio = _safe_div(turnover, abs(starting_equity))
    time_in_market = exposure

    eq_daily = eq.set_index("ts_utc")["equity"].resample("1D").last().dropna()
    daily_ret = eq_daily.pct_change().dropna()
    worst_1d = float(daily_ret.min()) if not daily_ret.empty else 0.0

    eq_weekly = eq.set_index("ts_utc")["equity"].resample("7D").last().dropna()
    weekly_ret = eq_weekly.pct_change().dropna()
    worst_1w = float(weekly_ret.min()) if not weekly_ret.empty else 0.0

    ret_sorted = np.sort(eq["ret"].to_numpy())
    q95 = np.percentile(ret_sorted, 95) if len(ret_sorted) > 0 else 0.0
    q05 = np.percentile(ret_sorted, 5) if len(ret_sorted) > 0 else 0.0
    tail_ratio = _safe_div(float(abs(q95)), float(abs(q05))) if q05 != 0 else 0.0

    dd_durations = []
    in_dd = False
    start_idx = 0
    for i, val in enumerate(drawdown.to_numpy()):
        if val < 0 and not in_dd:
            in_dd = True
            start_idx = i
        if val == 0 and in_dd:
            in_dd = False
            dd_durations.append(i - start_idx)
    if in_dd:
        dd_durations.append(len(drawdown) - start_idx)

    drawdown_duration = int(max(dd_durations)) if dd_durations else 0
    ulcer_index = float(np.sqrt(np.mean(np.square(drawdown.fillna(0.0).to_numpy()))))
    return_skewness = float(eq["ret"].skew()) if len(eq) > 2 else 0.0
    return_excess_kurtosis = float(eq["ret"].kurt()) if len(eq) > 3 else 0.0

    trades_per_day = float(trades_count / max((eq["ts_utc"].iloc[-1] - eq["ts_utc"].iloc[0]).days, 1))
    avg_bar_range = float(((eq["high"] - eq["low"]) / eq["close"].replace(0, np.nan)).mean()) if {"high", "low", "close"}.issubset(eq.columns) else 0.0

    return {
        "ok": True,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "trades_count": trades_count,
        "exposure": exposure,
        "turnover": turnover,
        "turnover_ratio": turnover_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "time_in_market": time_in_market,
        "worst_1d": worst_1d,
        "worst_1w": worst_1w,
        "tail_ratio": tail_ratio,
        "drawdown_duration": drawdown_duration,
        "ulcer_index": ulcer_index,
        "return_skewness": return_skewness,
        "return_excess_kurtosis": return_excess_kurtosis,
        "trades_per_day": trades_per_day,
        "avg_bar_range": avg_bar_range,
    }
