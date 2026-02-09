"""
Fill simulation — models realistic order execution.

Market orders: fill at bid/ask + slippage (always fill).
Limit orders: fill only if price trades through level (probabilistic).
"""

import random
from dataclasses import dataclass
from typing import Optional

from backtest.costs import CostConfig, slippage_pips


@dataclass
class FillResult:
    filled: bool
    fill_price: float = 0.0
    filled_qty: float = 0.0
    slippage_pips: float = 0.0


def simulate_market_fill(side: str, bid: float, ask: float,
                         qty: float, atr_pips: float,
                         cfg: CostConfig) -> FillResult:
    """
    Market order: always fills at bid/ask + slippage.

    BUY  -> fill at ask + slippage
    SELL -> fill at bid - slippage
    """
    slip = slippage_pips(atr_pips, cfg)
    slip_price = slip * cfg.pip_size

    if side in ("BUY", "LONG"):
        fill_price = ask + slip_price
    else:
        fill_price = bid - slip_price

    return FillResult(
        filled=True,
        fill_price=fill_price,
        filled_qty=qty,
        slippage_pips=slip,
    )


def simulate_limit_fill(side: str, order_price: float,
                        bar_high: float, bar_low: float,
                        qty: float,
                        cfg: CostConfig) -> FillResult:
    """
    Limit order: fills only if price trades through the order level.

    Fill probability depends on how far price traded through the level.
    Capped at limit_fill_rate_cap (default 85%).

    For BUY limit: order_price <= market, fills when bar_low <= order_price.
    For SELL limit: order_price >= market, fills when bar_high >= order_price.
    """
    if side in ("BUY", "LONG"):
        if bar_low > order_price:
            return FillResult(filled=False)

        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return FillResult(filled=False)

        distance_through = (order_price - bar_low) / bar_range
        fill_prob = min(cfg.limit_fill_rate_cap, 0.5 + distance_through * 0.5)

        if random.random() < fill_prob:
            # Partial fill: between 70% and 100% of requested qty
            filled_qty = qty * random.uniform(0.7, 1.0)
            return FillResult(
                filled=True,
                fill_price=order_price,
                filled_qty=filled_qty,
                slippage_pips=0.0,
            )
        return FillResult(filled=False)

    else:  # SELL limit
        if bar_high < order_price:
            return FillResult(filled=False)

        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return FillResult(filled=False)

        distance_through = (bar_high - order_price) / bar_range
        fill_prob = min(cfg.limit_fill_rate_cap, 0.5 + distance_through * 0.5)

        if random.random() < fill_prob:
            filled_qty = qty * random.uniform(0.7, 1.0)
            return FillResult(
                filled=True,
                fill_price=order_price,
                filled_qty=filled_qty,
                slippage_pips=0.0,
            )
        return FillResult(filled=False)
