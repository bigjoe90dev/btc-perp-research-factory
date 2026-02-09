"""
Position sizing strategies.

Supports:
- Fixed fractional (risk X% per trade)
- Kelly criterion (1/4 Kelly for safety)
- Fixed lot sizing
- Drawdown-based position reduction
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SizingConfig:
    method: str = "fixed_fractional"   # "fixed_fractional", "kelly", "fixed"
    risk_per_trade: float = 0.01       # 1% of account per trade
    fixed_lots: float = 0.1            # for "fixed" method
    max_margin_utilization: float = 0.9  # max 90% of account as margin
    leverage: float = 30.0             # typical forex leverage

    # Kelly criterion inputs (updated from running stats)
    kelly_win_rate: float = 0.50
    kelly_avg_win_pips: float = 20.0
    kelly_avg_loss_pips: float = 15.0
    kelly_fraction: float = 0.25       # use 1/4 Kelly

    # Drawdown circuit breakers
    dd_reduce_threshold: float = 0.10  # reduce to 50% size at 10% DD
    dd_halt_threshold: float = 0.15    # stop trading at 15% DD

    # Instrument
    pip_value_per_lot: float = 10.0    # USD per pip per standard lot


def calculate_position_size(
    account_balance: float,
    stop_loss_pips: float,
    current_drawdown: float = 0.0,
    cfg: SizingConfig = None,
) -> float:
    """
    Calculate position size in standard lots.

    Args:
        account_balance: Current account balance (USD)
        stop_loss_pips: Distance to stop loss in pips
        current_drawdown: Current drawdown as fraction (e.g., 0.05 = 5%)
        cfg: Sizing configuration

    Returns:
        Position size in standard lots (e.g., 0.1 = mini lot)
    """
    if cfg is None:
        cfg = SizingConfig()

    # Check circuit breakers first
    action = drawdown_action(current_drawdown, cfg)
    if action == "HALT":
        return 0.0

    if cfg.method == "fixed":
        lots = cfg.fixed_lots

    elif cfg.method == "kelly":
        lots = _kelly_size(account_balance, stop_loss_pips, cfg)

    else:  # fixed_fractional (default)
        lots = _fixed_fractional_size(account_balance, stop_loss_pips, cfg)

    # Apply drawdown reduction
    if action == "REDUCE":
        lots *= 0.5

    # Enforce margin limits
    lots = _enforce_margin_limit(lots, account_balance, cfg)

    # Floor to micro-lot (0.01)
    lots = max(0.01, round(lots, 2))

    return lots


def _fixed_fractional_size(balance: float, sl_pips: float,
                           cfg: SizingConfig) -> float:
    """Risk X% of account per trade."""
    if sl_pips <= 0:
        return cfg.fixed_lots  # fallback

    risk_amount = balance * cfg.risk_per_trade
    lots = risk_amount / (sl_pips * cfg.pip_value_per_lot)
    return lots


def _kelly_size(balance: float, sl_pips: float,
                cfg: SizingConfig) -> float:
    """Kelly criterion position sizing (conservative 1/4 Kelly)."""
    w = cfg.kelly_win_rate
    avg_w = cfg.kelly_avg_win_pips
    avg_l = cfg.kelly_avg_loss_pips

    if avg_w <= 0:
        return cfg.fixed_lots

    kelly_f = (w * avg_w - (1 - w) * avg_l) / avg_w
    safe_f = max(0, kelly_f * cfg.kelly_fraction)

    risk_amount = balance * safe_f
    if sl_pips > 0:
        lots = risk_amount / (sl_pips * cfg.pip_value_per_lot)
    else:
        lots = cfg.fixed_lots

    return lots


def _enforce_margin_limit(lots: float, balance: float,
                          cfg: SizingConfig) -> float:
    """Cap position to stay within margin utilization limits."""
    # 1 standard lot = 100,000 units
    notional = lots * 100_000
    margin_required = notional / cfg.leverage
    max_margin = balance * cfg.max_margin_utilization

    if margin_required > max_margin:
        lots = (max_margin * cfg.leverage) / 100_000

    return lots


def drawdown_action(current_drawdown: float,
                    cfg: SizingConfig = None) -> str:
    """
    Determine action based on current drawdown level.

    Returns: "NORMAL", "REDUCE", or "HALT"
    """
    if cfg is None:
        cfg = SizingConfig()

    if current_drawdown >= cfg.dd_halt_threshold:
        return "HALT"
    elif current_drawdown >= cfg.dd_reduce_threshold:
        return "REDUCE"
    return "NORMAL"
