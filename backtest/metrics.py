"""
Performance metrics for backtest evaluation.

Computes: CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor,
expectancy, Calmar ratio, and more.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class BacktestMetrics:
    # Returns
    cagr: float = 0.0
    total_return_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Costs
    total_spread_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_commission_cost: float = 0.0
    total_swap_cost: float = 0.0
    total_costs: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "=== Backtest Results ===",
            f"  CAGR:           {self.cagr:.2%}",
            f"  Total Return:   {self.total_return_pct:.2%}",
            f"  Sharpe:         {self.sharpe_ratio:.2f}",
            f"  Sortino:        {self.sortino_ratio:.2f}",
            f"  Calmar:         {self.calmar_ratio:.2f}",
            f"  Max Drawdown:   {self.max_drawdown_pct:.2%}",
            f"  DD Duration:    {self.max_drawdown_duration_bars} bars",
            "",
            f"  Trades:         {self.total_trades}",
            f"  Win Rate:       {self.win_rate:.1%}",
            f"  Avg Win:        ${self.avg_win:.2f}",
            f"  Avg Loss:       ${self.avg_loss:.2f}",
            f"  W/L Ratio:      {self.avg_win_loss_ratio:.2f}",
            f"  Profit Factor:  {self.profit_factor:.2f}",
            f"  Expectancy:     ${self.expectancy:.2f}",
            "",
            f"  Total Costs:    ${self.total_costs:.2f}",
            f"    Spread:       ${self.total_spread_cost:.2f}",
            f"    Slippage:     ${self.total_slippage_cost:.2f}",
            f"    Commission:   ${self.total_commission_cost:.2f}",
            f"    Swap:         ${self.total_swap_cost:.2f}",
        ]
        return "\n".join(lines)


def compute_metrics(equity_curve: np.ndarray,
                    trade_pnls: List[float],
                    trade_costs: Optional[dict] = None,
                    bars_per_year: float = 252 * 86400,
                    risk_free_rate: float = 0.0) -> BacktestMetrics:
    """
    Compute all performance metrics from an equity curve and trade P&Ls.

    Args:
        equity_curve: Array of equity values at each bar
        trade_pnls: List of P&L for each closed trade (after costs)
        trade_costs: Dict with keys: spread, slippage, commission, swap (totals)
        bars_per_year: Number of bars in a year (for annualization)
        risk_free_rate: Annual risk-free rate (default 0)
    """
    m = BacktestMetrics()

    if len(equity_curve) < 2 or len(trade_pnls) == 0:
        return m

    # --- Returns ---
    initial = equity_curve[0]
    final = equity_curve[-1]
    m.total_return_pct = (final - initial) / initial if initial > 0 else 0.0

    # CAGR
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    if years > 0 and initial > 0 and final > 0:
        m.cagr = (final / initial) ** (1 / years) - 1
    else:
        m.cagr = 0.0

    # --- Bar-level returns for Sharpe/Sortino ---
    eq = np.array(equity_curve, dtype=float)
    bar_returns = np.diff(eq) / eq[:-1]
    bar_returns = bar_returns[np.isfinite(bar_returns)]

    if len(bar_returns) > 1:
        mean_ret = np.mean(bar_returns)
        std_ret = np.std(bar_returns, ddof=1)
        annualization = math.sqrt(bars_per_year)

        # Sharpe
        if std_ret > 0:
            m.sharpe_ratio = (mean_ret / std_ret) * annualization
        else:
            m.sharpe_ratio = 0.0

        # Sortino (downside deviation only)
        downside = bar_returns[bar_returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            if downside_std > 0:
                m.sortino_ratio = (mean_ret / downside_std) * annualization

    # --- Drawdown ---
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    m.max_drawdown_pct = abs(float(np.min(drawdowns)))

    # Drawdown duration (longest streak below peak)
    in_dd = eq < running_max
    max_dd_dur = 0
    current_dur = 0
    for below in in_dd:
        if below:
            current_dur += 1
            max_dd_dur = max(max_dd_dur, current_dur)
        else:
            current_dur = 0
    m.max_drawdown_duration_bars = max_dd_dur

    # Calmar
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.cagr / m.max_drawdown_pct

    # --- Trade stats ---
    pnls = np.array(trade_pnls, dtype=float)
    m.total_trades = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    m.winning_trades = len(wins)
    m.losing_trades = len(losses)
    m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0

    m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    m.avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    if m.avg_loss > 0:
        m.avg_win_loss_ratio = m.avg_win / m.avg_loss
    else:
        m.avg_win_loss_ratio = float("inf") if m.avg_win > 0 else 0.0

    # Profit factor
    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
    if gross_losses > 0:
        m.profit_factor = gross_wins / gross_losses
    else:
        m.profit_factor = float("inf") if gross_wins > 0 else 0.0

    # Expectancy
    m.expectancy = float(np.mean(pnls))

    # --- Costs ---
    if trade_costs:
        m.total_spread_cost = trade_costs.get("spread", 0.0)
        m.total_slippage_cost = trade_costs.get("slippage", 0.0)
        m.total_commission_cost = trade_costs.get("commission", 0.0)
        m.total_swap_cost = trade_costs.get("swap", 0.0)
    m.total_costs = (m.total_spread_cost + m.total_slippage_cost +
                     m.total_commission_cost + m.total_swap_cost)

    return m
