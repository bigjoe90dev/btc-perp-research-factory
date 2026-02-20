from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

from .types import Bar, Fill


def _vol_proxy(bar: Bar, mode: str) -> float:
    if mode == "candle_range":
        if bar.open <= 0:
            return 0.0
        return max((bar.high - bar.low) / bar.open, 0.0)
    if mode == "atr_like":
        mid = (bar.high + bar.low + bar.close + bar.open) / 4.0
        if mid <= 0:
            return 0.0
        return max((bar.high - bar.low) / mid, 0.0)
    return 0.0


def compute_slippage_bps(bar: Bar, slippage_cfg: dict[str, Any], execution_cfg: dict[str, Any]) -> float:
    base = float(slippage_cfg.get("bps_base", 0.0))
    per_vol = float(slippage_cfg.get("bps_per_vol", 0.0))
    mode = str(slippage_cfg.get("vol_proxy", "candle_range"))
    proxy = _vol_proxy(bar, mode)

    # Approximate on-chain/sequencer timing cost by adding adverse slippage with delay.
    delay_sec = float(
        execution_cfg.get(
            "execution_delay_seconds",
            execution_cfg.get("latency_seconds", 0.0),
        )
    )
    delay_bps_per_second = float(execution_cfg.get("delay_bps_per_second", 0.0))
    delay_bps_per_vol_per_second = float(execution_cfg.get("delay_bps_per_vol_per_second", 0.0))
    delay_bps = delay_sec * (delay_bps_per_second + delay_bps_per_vol_per_second * proxy)
    return max(base + per_vol * proxy + delay_bps, 0.0)


def round_qty_to_step(qty_abs: float, qty_step: float) -> float:
    if qty_step <= 0:
        return float(max(qty_abs, 0.0))
    return float(math.floor(max(qty_abs, 0.0) / qty_step) * qty_step)


def round_price_to_tick(price: float, tick_size: float, side: int) -> float:
    if tick_size <= 0:
        return float(price)
    tick = Decimal(str(tick_size))
    px = Decimal(str(price))
    units = px / tick
    # Conservative rounding: pay more on buys, receive less on sells.
    rounding = ROUND_CEILING if side > 0 else ROUND_FLOOR
    units_i = units.to_integral_value(rounding=rounding)
    rounded = units_i * tick
    if rounded < tick:
        rounded = tick
    return float(rounded)


def execute_market_order_next_open(
    next_bar: Bar,
    qty_signed: float,
    reason: str,
    fee_bps: float,
    slippage_cfg: dict[str, Any],
    execution_cfg: dict[str, Any],
    slippage_bar: Bar,
) -> tuple[Fill, float]:
    """
    Returns (fill, cash_delta) where cash_delta is applied to account cash.
    """
    if qty_signed == 0:
        raise ValueError("qty_signed must be non-zero")

    # Use a bar known at decision time for volatility-dependent slippage terms.
    slip_bps = compute_slippage_bps(slippage_bar, slippage_cfg, execution_cfg=execution_cfg)
    side = 1 if qty_signed > 0 else -1

    if side > 0:
        raw_price = next_bar.open * (1.0 + slip_bps / 10_000.0)
    else:
        raw_price = next_bar.open * (1.0 - slip_bps / 10_000.0)

    price_tick_size = float(execution_cfg.get("price_tick_size", 0.0))
    price = round_price_to_tick(raw_price, price_tick_size, side=side)

    notional = abs(qty_signed) * price
    fee_paid = notional * (fee_bps / 10_000.0)

    # Buy decreases cash. Sell increases cash.
    cash_delta = -qty_signed * price - fee_paid

    fill = Fill(
        ts_utc=next_bar.ts_utc,
        side=side,
        qty=abs(float(qty_signed)),
        price=float(price),
        fee_paid=float(fee_paid),
        slippage_bps=float(slip_bps),
        reason=reason,
    )
    return fill, float(cash_delta)
