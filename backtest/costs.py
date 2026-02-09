"""
Realistic forex transaction cost models.

Covers:
- Spread cost (bid-ask)
- Broker commission (per-lot)
- Slippage (fixed + volatility-based)
- Swap rates (overnight financing)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """All cost parameters for a backtest run. Forex / cTrader defaults for EURUSD."""

    # Spread — usually captured from live bid/ask, but this is a fallback
    spread_typical_pips: float = 1.2
    spread_news_pips: float = 5.0
    spread_min_pips: float = 0.1
    spread_max_pips: float = 10.0

    # Session multipliers
    asia_spread_mult: float = 3.0
    frankfurt_spread_mult: float = 1.6
    london_spread_mult: float = 1.2
    newyork_spread_mult: float = 1.4
    off_spread_mult: float = 2.0

    # Broker commission (per standard lot per side, USD)
    commission_per_lot_per_side: float = 3.50

    # Slippage
    slippage_base_pips: float = 0.1       # fixed minimum
    slippage_vol_factor: float = 0.05     # multiplied by recent ATR in pips
    asia_slip_mult: float = 1.2
    frankfurt_slip_mult: float = 1.1
    london_slip_mult: float = 1.3
    newyork_slip_mult: float = 1.2
    off_slip_mult: float = 1.4

    # Swap rates (pips per day, negative = cost)
    swap_long_pips_per_day: float = -0.5
    swap_short_pips_per_day: float = -0.3
    triple_swap_day: int = 2              # Wednesday (0=Mon, 2=Wed)

    # Instrument constants
    pip_size: float = 0.0001              # EURUSD = 4th decimal
    pip_value_per_lot: float = 10.0       # USD per pip per standard lot

    # Fill simulation
    limit_fill_rate_cap: float = 0.85     # max probability a limit order fills

    # Margin
    margin_stop_out_level: float = 0.50   # 50% margin level = forced close


def spread_cost_usd(bid: float, ask: float, lots: float,
                    cfg: CostConfig) -> float:
    """Half-spread cost applied to both entry and exit (full round-trip)."""
    spread_pips = (ask - bid) / cfg.pip_size
    return spread_pips * cfg.pip_value_per_lot * lots


def commission_cost_usd(lots: float, cfg: CostConfig) -> float:
    """Round-trip commission."""
    return cfg.commission_per_lot_per_side * 2 * lots


def slippage_pips(atr_pips: float, cfg: CostConfig) -> float:
    """Estimated slippage in pips based on fixed base + volatility component."""
    return cfg.slippage_base_pips + (cfg.slippage_vol_factor * atr_pips)


def session_spread_multiplier(session: str, cfg: CostConfig) -> float:
    s = (session or "Off").lower()
    if s == "asia":
        return cfg.asia_spread_mult
    if s == "frankfurt":
        return cfg.frankfurt_spread_mult
    if s == "london":
        return cfg.london_spread_mult
    if s == "newyork":
        return cfg.newyork_spread_mult
    return cfg.off_spread_mult


def session_slip_multiplier(session: str, cfg: CostConfig) -> float:
    s = (session or "Off").lower()
    if s == "asia":
        return cfg.asia_slip_mult
    if s == "frankfurt":
        return cfg.frankfurt_slip_mult
    if s == "london":
        return cfg.london_slip_mult
    if s == "newyork":
        return cfg.newyork_slip_mult
    return cfg.off_slip_mult


def clamp_spread(spread_pips: float, cfg: CostConfig) -> float:
    return max(cfg.spread_min_pips, min(cfg.spread_max_pips, spread_pips))


def slippage_cost_usd(atr_pips: float, lots: float,
                      cfg: CostConfig) -> float:
    """Slippage cost in USD for a round-trip (entry + exit)."""
    slip_pips = slippage_pips(atr_pips, cfg)
    return slip_pips * 2 * cfg.pip_value_per_lot * lots


def swap_cost_usd(side: str, lots: float, days_held: float,
                  weekday_at_entry: int, cfg: CostConfig) -> float:
    """
    Swap (overnight financing) cost.

    Only charged if position crosses the daily rollover (17:00 EST / 22:00 UTC).
    Wednesday = triple swap to cover the weekend.

    Args:
        side: "LONG" or "SHORT"
        lots: position size in standard lots
        days_held: number of calendar days the position was held
        weekday_at_entry: 0=Monday .. 6=Sunday
        cfg: cost configuration
    """
    if days_held < 1:
        return 0.0  # intraday, no swap

    rate = cfg.swap_long_pips_per_day if side == "LONG" else cfg.swap_short_pips_per_day

    # Count how many rollovers we crossed
    swap_days = int(days_held)

    # Count Wednesdays in the holding period (+2 extra days for triple swap)
    extra_from_triple = 0
    for d in range(swap_days):
        day_of_week = (weekday_at_entry + d) % 7
        if day_of_week == cfg.triple_swap_day:
            extra_from_triple += 2  # triple = 2 extra

    effective_days = swap_days + extra_from_triple
    return abs(rate) * cfg.pip_value_per_lot * lots * effective_days


def total_trade_cost_usd(bid: float, ask: float, lots: float,
                         atr_pips: float, side: str, days_held: float,
                         weekday_at_entry: int, cfg: CostConfig) -> float:
    """Total cost for a single round-trip trade."""
    spread = spread_cost_usd(bid, ask, lots, cfg)
    commission = commission_cost_usd(lots, cfg)
    slippage = slippage_cost_usd(atr_pips, lots, cfg)
    swap = swap_cost_usd(side, lots, days_held, weekday_at_entry, cfg)
    return spread + commission + slippage + swap


def apply_fill_price(side: str, bid: float, ask: float,
                     atr_pips: float, cfg: CostConfig) -> float:
    """
    Compute the actual fill price for a market order (including slippage).

    BUY fills at ask + slippage.
    SELL fills at bid - slippage.
    """
    slip = slippage_pips(atr_pips, cfg) * cfg.pip_size

    if side in ("BUY", "LONG"):
        return ask + slip
    else:  # SELL / SHORT
        return bid - slip
