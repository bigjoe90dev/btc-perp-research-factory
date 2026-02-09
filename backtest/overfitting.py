"""
Overfitting detection.

Splits data into train/test, compares performance, flags overfitting.
Computes composite overfitting score (0-100).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from backtest.engine import BacktestEngine, MACrossStrategy, Strategy
from backtest.costs import CostConfig


@dataclass
class OverfitReport:
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_profit_factor: float = 0.0
    test_profit_factor: float = 0.0
    train_max_drawdown: float = 0.0
    test_max_drawdown: float = 0.0
    train_trades: int = 0
    test_trades: int = 0

    sharpe_degradation: float = 0.0       # (train - test) / train
    pf_degradation: float = 0.0
    dd_increase: float = 0.0              # (test_dd - train_dd) / train_dd

    flags: List[str] = field(default_factory=list)
    is_overfit: bool = False

    # Composite score: 0 = definitely overfit, 100 = very robust
    robustness_score: float = 0.0

    def summary(self) -> str:
        status = "OVERFIT" if self.is_overfit else "OK"
        lines = [
            f"=== Overfitting Analysis: {status} (score: {self.robustness_score:.0f}/100) ===",
            f"                Train      Test       Degradation",
            f"  Sharpe:       {self.train_sharpe:>8.2f}   {self.test_sharpe:>8.2f}   {self.sharpe_degradation:>8.0%}",
            f"  Profit F:     {self.train_profit_factor:>8.2f}   {self.test_profit_factor:>8.2f}   {self.pf_degradation:>8.0%}",
            f"  Max DD:       {self.train_max_drawdown:>8.2%}   {self.test_max_drawdown:>8.2%}   {self.dd_increase:>+8.0%}",
            f"  Trades:       {self.train_trades:>8}   {self.test_trades:>8}",
        ]
        if self.flags:
            lines.append("")
            lines.append("  Flags:")
            for f in self.flags:
                lines.append(f"    - {f}")
        return "\n".join(lines)


def detect_overfitting(
    df: pd.DataFrame,
    strategy_params: dict = None,
    split_ratio: float = 0.7,
    cost_config: CostConfig = None,
    initial_balance: float = 10_000.0,
    size_lots: float = 0.1,
    symbol: str = "EURUSD",
    sharpe_deg_threshold: float = 0.30,
    pf_deg_threshold: float = 0.30,
    dd_increase_threshold: float = 0.50,
    strategy_factory=None,
    engine_kwargs: Optional[dict] = None,
) -> OverfitReport:
    """
    Split data into train/test, run backtest on both, compare performance.

    Args:
        df: DataFrame with candle columns
        strategy_params: Dict of strategy params (default: {"fast_n": 10, "slow_n": 30})
        split_ratio: Fraction of data used for training (default 70%)
        sharpe_deg_threshold: Flag if Sharpe degrades by more than this (30%)
        pf_deg_threshold: Flag if profit factor degrades more than this
        dd_increase_threshold: Flag if max DD increases more than this
    """
    cfg = cost_config or CostConfig()
    params = strategy_params or {"fast_n": 10, "slow_n": 30}
    engine_kwargs = engine_kwargs or {}
    factory = strategy_factory or (lambda **p: MACrossStrategy(**p))

    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["ts_utc"]):
        data["ts_utc"] = pd.to_datetime(data["ts_utc"])
    data = data.sort_values("ts_utc").reset_index(drop=True)

    split_idx = int(len(data) * split_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    # Run on train
    train_strat: Strategy = factory(**params)
    train_engine = BacktestEngine(train_strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs)
    train_result = train_engine.run(train_df)

    # Run on test
    test_strat: Strategy = factory(**params)
    test_engine = BacktestEngine(test_strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs)
    test_result = test_engine.run(test_df)

    report = OverfitReport(
        train_sharpe=train_result.metrics.sharpe_ratio,
        test_sharpe=test_result.metrics.sharpe_ratio,
        train_profit_factor=train_result.metrics.profit_factor,
        test_profit_factor=test_result.metrics.profit_factor,
        train_max_drawdown=train_result.metrics.max_drawdown_pct,
        test_max_drawdown=test_result.metrics.max_drawdown_pct,
        train_trades=train_result.metrics.total_trades,
        test_trades=test_result.metrics.total_trades,
    )

    # Degradation calculations
    if abs(report.train_sharpe) > 0.01:
        report.sharpe_degradation = (
            (report.train_sharpe - report.test_sharpe) / abs(report.train_sharpe)
        )
    if report.train_profit_factor > 0.01:
        report.pf_degradation = (
            (report.train_profit_factor - report.test_profit_factor) / report.train_profit_factor
        )
    if report.train_max_drawdown > 0.001:
        report.dd_increase = (
            (report.test_max_drawdown - report.train_max_drawdown) / report.train_max_drawdown
        )

    # Flags
    if report.sharpe_degradation > sharpe_deg_threshold:
        report.flags.append(f"Sharpe degraded {report.sharpe_degradation:.0%} (threshold {sharpe_deg_threshold:.0%})")
    if report.pf_degradation > pf_deg_threshold:
        report.flags.append(f"Profit factor degraded {report.pf_degradation:.0%}")
    if report.dd_increase > dd_increase_threshold:
        report.flags.append(f"Max drawdown increased {report.dd_increase:.0%}")
    if report.test_trades < 10:
        report.flags.append(f"Too few OOS trades ({report.test_trades})")
    if report.test_sharpe < 0:
        report.flags.append("Negative OOS Sharpe")

    report.is_overfit = len(report.flags) > 0

    # Composite robustness score (0-100)
    score = 100.0
    score -= min(40, max(0, report.sharpe_degradation * 100))  # up to -40 for degradation
    score -= min(20, max(0, report.pf_degradation * 50))       # up to -20 for PF degradation
    score -= min(20, max(0, report.dd_increase * 40))          # up to -20 for DD increase
    if report.test_trades < 30:
        score -= 10  # penalty for small sample
    if report.test_sharpe < 0:
        score -= 10  # penalty for negative OOS
    report.robustness_score = max(0, min(100, score))

    return report
