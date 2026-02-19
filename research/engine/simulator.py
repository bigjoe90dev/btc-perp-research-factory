from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.data.funding import funding_cashflow
from research.strategies.base import Strategy, StrategyContext

from .execution import execute_market_order_next_open, round_qty_to_step
from .types import Bar, Fill, FundingEvent, SimulationResult, Trade


def _to_bars(df: pd.DataFrame) -> list[Bar]:
    return [
        Bar(
            ts_utc=pd.Timestamp(r.ts_utc),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
        )
        for r in df.itertuples(index=False)
    ]


def _bars_per_year(timeframe: str, cfg: dict[str, Any]) -> int:
    metrics_cfg = cfg.get("metrics", {})
    if timeframe == "5m":
        return int(metrics_cfg.get("annualization_bars_5m", 105_120))
    if timeframe == "1h":
        return int(metrics_cfg.get("annualization_bars_1h", 8_760))
    return 525_600


def _side_from_qty(qty: float, eps: float = 1e-12) -> int:
    if qty > eps:
        return 1
    if qty < -eps:
        return -1
    return 0


def run_backtest(
    candles: pd.DataFrame,
    funding_rates_by_bar: np.ndarray,
    strategy: Strategy,
    timeframe: str,
    backtest_cfg: dict[str, Any],
) -> SimulationResult:
    if len(candles) < 3:
        raise RuntimeError("Not enough candles for simulation")
    if len(candles) != len(funding_rates_by_bar):
        raise ValueError("funding_rates_by_bar length mismatch")

    bars = _to_bars(candles)
    exec_cfg = backtest_cfg.get("execution", {})
    size_cfg = backtest_cfg.get("position_sizing", {})

    initial_equity = float(backtest_cfg.get("initial_equity", 10_000.0))
    leverage = float(size_cfg.get("leverage", 1.0))
    risk_fraction = float(size_cfg.get("risk_fraction", 1.0))

    fee_bps = float(exec_cfg.get("taker_fee_bps", 0.0))
    slippage_cfg = dict(exec_cfg.get("slippage", {}))
    qty_step = float(exec_cfg.get("qty_step", 0.0))
    min_notional = float(exec_cfg.get("min_notional_usd", 0.0))
    max_pos_notional = float(exec_cfg.get("max_position_notional_usd", 0.0))
    max_participation = float(exec_cfg.get("max_candle_volume_participation", 1.0))
    participation_mode = str(exec_cfg.get("participation_mode", "reject")).lower()
    if participation_mode not in {"reject", "partial"}:
        participation_mode = "reject"

    cash = initial_equity
    position_qty = 0.0
    current_side = 0

    fills: list[Fill] = []
    funding_events: list[FundingEvent] = []
    trades: list[Trade] = []

    open_trade: dict[str, Any] | None = None
    fill_notional_by_idx: dict[int, float] = {}
    skipped_participation = 0
    skipped_min_notional = 0
    skipped_small_qty = 0
    partial_fills = 0

    rows: list[dict[str, Any]] = []
    warmup = int(strategy.warmup_bars())

    for i, bar in enumerate(bars):
        rate = float(funding_rates_by_bar[i])
        if rate != 0.0 and position_qty != 0.0:
            cf = funding_cashflow(position_qty=position_qty, mark_price=bar.close, funding_rate=rate)
            cash += cf
            funding_events.append(FundingEvent(ts_utc=bar.ts_utc, rate=rate, cashflow=cf))
            if open_trade is not None:
                open_trade["funding"] += cf

        equity = cash + position_qty * bar.close

        rows.append(
            {
                "ts_utc": bar.ts_utc,
                "equity": float(equity),
                "cash": float(cash),
                "position_qty": float(position_qty),
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "gross_notional": float(abs(position_qty) * bar.close),
                "trade_notional": float(fill_notional_by_idx.get(i, 0.0)),
            }
        )

        if i >= len(bars) - 1:
            continue
        if i < warmup:
            continue

        ctx = StrategyContext(candles, i)
        signal = strategy.on_bar(ctx=ctx, current_side=current_side)
        target_side = int(max(-1, min(1, signal.target_side)))

        if target_side == current_side:
            continue

        # Use equity at decision time for x1 notional sizing.
        target_notional = max(equity * risk_fraction * leverage, 0.0)
        if max_pos_notional > 0.0:
            target_notional = min(target_notional, max_pos_notional)
        next_bar = bars[i + 1]
        if next_bar.open <= 0:
            continue
        target_qty = 0.0 if target_side == 0 else (target_notional / next_bar.open) * float(target_side)

        qty_delta_requested = float(target_qty - position_qty)
        if abs(qty_delta_requested) < 1e-12:
            current_side = target_side
            continue

        qty_delta = qty_delta_requested
        if max_participation >= 0.0:
            max_qty_for_bar = max(next_bar.volume, 0.0) * max_participation
            if abs(qty_delta) > max_qty_for_bar:
                if participation_mode == "reject":
                    skipped_participation += 1
                    continue
                qty_delta = float(np.sign(qty_delta) * max_qty_for_bar)
                partial_fills += 1

        qty_abs_rounded = round_qty_to_step(abs(qty_delta), qty_step)
        if qty_abs_rounded <= 0.0:
            skipped_small_qty += 1
            continue
        qty_delta = float(np.sign(qty_delta) * qty_abs_rounded)

        if min_notional > 0.0 and (abs(qty_delta) * next_bar.open) < min_notional:
            skipped_min_notional += 1
            continue

        fill, cash_delta = execute_market_order_next_open(
            next_bar=next_bar,
            qty_signed=qty_delta,
            reason=signal.reason,
            fee_bps=fee_bps,
            slippage_cfg=slippage_cfg,
            execution_cfg=exec_cfg,
        )
        fills.append(fill)
        cash += cash_delta
        fill_notional_by_idx[i + 1] = fill_notional_by_idx.get(i + 1, 0.0) + abs(qty_delta) * fill.price

        prev_qty = position_qty
        prev_side = _side_from_qty(prev_qty)
        position_qty = float(position_qty + qty_delta)
        new_side = _side_from_qty(position_qty)
        current_side = new_side

        prev_abs = abs(prev_qty)
        new_abs = abs(position_qty)
        total_abs = abs(qty_delta)

        if prev_side == 0 and new_side != 0:
            open_trade = {
                "entry_ts_utc": fill.ts_utc,
                "entry_idx": i + 1,
                "entry_price": float(fill.price),
                "qty": float(new_abs),
                "side": int(new_side),
                "entry_fee": float(fill.fee_paid),
                "funding": 0.0,
            }
            continue

        if prev_side == 0:
            continue

        if open_trade is None:
            open_trade = {
                "entry_ts_utc": bars[max(i, 0)].ts_utc,
                "entry_idx": i,
                "entry_price": float(bars[i].close),
                "qty": float(prev_abs),
                "side": int(prev_side),
                "entry_fee": 0.0,
                "funding": 0.0,
            }

        if new_side == prev_side:
            if new_abs > prev_abs:
                add_qty = new_abs - prev_abs
                old_qty = float(open_trade["qty"])
                total_qty = old_qty + add_qty
                if total_qty > 0:
                    open_trade["entry_price"] = (
                        float(open_trade["entry_price"]) * old_qty + float(fill.price) * add_qty
                    ) / total_qty
                open_trade["qty"] = float(total_qty)
                open_trade["entry_fee"] = float(open_trade["entry_fee"]) + float(fill.fee_paid)
            elif new_abs < prev_abs:
                close_qty = prev_abs - new_abs
                close_ratio = close_qty / max(float(open_trade["qty"]), 1e-12)
                entry_fee_alloc = float(open_trade["entry_fee"]) * close_ratio
                funding_alloc = float(open_trade["funding"]) * close_ratio

                side = int(open_trade["side"])
                entry = float(open_trade["entry_price"])
                exit_px = float(fill.price)
                pnl = (exit_px - entry) * close_qty if side > 0 else (entry - exit_px) * close_qty
                pnl_after = pnl - entry_fee_alloc - float(fill.fee_paid) + funding_alloc
                trades.append(
                    Trade(
                        entry_ts_utc=pd.Timestamp(open_trade["entry_ts_utc"]),
                        exit_ts_utc=fill.ts_utc,
                        side=side,
                        entry_price=entry,
                        exit_price=exit_px,
                        qty=close_qty,
                        pnl=pnl,
                        pnl_after_costs=pnl_after,
                        bars_held=max(int(i - int(open_trade["entry_idx"])), 1),
                    )
                )

                open_trade["qty"] = float(new_abs)
                open_trade["entry_fee"] = float(open_trade["entry_fee"]) - entry_fee_alloc
                open_trade["funding"] = float(open_trade["funding"]) - funding_alloc
                if new_abs <= 1e-12:
                    open_trade = None
            continue

        # Side flip (can include full close and reverse in a single fill).
        close_qty = prev_abs
        open_qty = new_abs
        fee_close = float(fill.fee_paid) * (close_qty / total_abs) if total_abs > 0 else 0.0
        fee_open = float(fill.fee_paid) * (open_qty / total_abs) if total_abs > 0 else 0.0

        side = int(open_trade["side"])
        entry = float(open_trade["entry_price"])
        exit_px = float(fill.price)
        pnl = (exit_px - entry) * close_qty if side > 0 else (entry - exit_px) * close_qty
        pnl_after = pnl - float(open_trade["entry_fee"]) - fee_close + float(open_trade["funding"])
        trades.append(
            Trade(
                entry_ts_utc=pd.Timestamp(open_trade["entry_ts_utc"]),
                exit_ts_utc=fill.ts_utc,
                side=side,
                entry_price=entry,
                exit_price=exit_px,
                qty=close_qty,
                pnl=pnl,
                pnl_after_costs=pnl_after,
                bars_held=max(int(i - int(open_trade["entry_idx"])), 1),
            )
        )

        if new_side == 0:
            open_trade = None
        else:
            open_trade = {
                "entry_ts_utc": fill.ts_utc,
                "entry_idx": i + 1,
                "entry_price": float(fill.price),
                "qty": float(open_qty),
                "side": int(new_side),
                "entry_fee": float(fee_open),
                "funding": 0.0,
            }

    # Force-close open trade at final close for reporting completeness.
    if open_trade is not None:
        last = bars[-1]
        side = int(open_trade["side"])
        qty = float(open_trade["qty"])
        entry = float(open_trade["entry_price"])
        exit_px = float(last.close)
        pnl = (exit_px - entry) * qty if side > 0 else (entry - exit_px) * qty
        pnl_after = pnl - float(open_trade["entry_fee"]) + float(open_trade["funding"])
        trades.append(
            Trade(
                entry_ts_utc=pd.Timestamp(open_trade["entry_ts_utc"]),
                exit_ts_utc=last.ts_utc,
                side=side,
                entry_price=entry,
                exit_price=exit_px,
                qty=qty,
                pnl=pnl,
                pnl_after_costs=pnl_after,
                bars_held=max(int(len(bars) - 1 - open_trade["entry_idx"]), 1),
            )
        )

    eq = pd.DataFrame(rows)
    summary = {
        "bars": len(eq),
        "fills": len(fills),
        "trades": len(trades),
        "funding_events": len(funding_events),
        "skipped_participation": skipped_participation,
        "skipped_min_notional": skipped_min_notional,
        "skipped_small_qty": skipped_small_qty,
        "partial_fills": partial_fills,
        "bars_per_year": _bars_per_year(timeframe=timeframe, cfg=backtest_cfg),
    }
    return SimulationResult(
        equity_curve=eq,
        trades=trades,
        fills=fills,
        funding_events=funding_events,
        summary=summary,
    )
