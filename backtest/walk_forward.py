"""
Walk-forward analysis.

Splits data into rolling train/test windows, optimizes on training set,
evaluates on test set, and reports consistency across all windows.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, MACrossStrategy, Strategy
from backtest.costs import CostConfig
from backtest.metrics import BacktestMetrics


@dataclass
class WindowResult:
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_profit_factor: float
    test_total_trades: int
    degradation: float  # (train_sharpe - test_sharpe) / train_sharpe


@dataclass
class WalkForwardResult:
    windows: List[WindowResult]
    profitable_windows: int
    total_windows: int
    consistency_pct: float         # % of windows that are profitable OOS
    avg_oos_sharpe: float
    avg_degradation: float

    def summary(self) -> str:
        lines = [
            "=== Walk-Forward Analysis ===",
            f"  Windows:         {self.total_windows}",
            f"  Profitable OOS:  {self.profitable_windows} ({self.consistency_pct:.1f}%)",
            f"  Avg OOS Sharpe:  {self.avg_oos_sharpe:.2f}",
            f"  Avg Degradation: {self.avg_degradation:.1%}",
            "",
        ]
        for w in self.windows:
            status = "PASS" if w.test_profit_factor > 1.0 else "FAIL"
            lines.append(
                f"  [{w.window_idx}] {status} | "
                f"train={w.train_sharpe:.2f} test={w.test_sharpe:.2f} "
                f"deg={w.degradation:.0%} trades={w.test_total_trades}"
            )
        return "\n".join(lines)


def walk_forward_analysis(
    df: pd.DataFrame,
    train_days: int = 90,
    test_days: int = 30,
    step_days: int = 30,
    param_grid: Optional[List[dict]] = None,
    cost_config: CostConfig = None,
    initial_balance: float = 10_000.0,
    size_lots: float = 0.1,
    symbol: str = "EURUSD",
    strategy_factory=None,
    engine_kwargs: Optional[dict] = None,
) -> WalkForwardResult:
    """
    Run walk-forward analysis with rolling windows.

    Args:
        df: DataFrame with candle columns
        train_days: Length of training window in calendar days
        test_days: Length of test window in calendar days
        step_days: How far to slide the window each iteration
        param_grid: List of param dicts (e.g. [{"fast_n": 5, "slow_n": 20}, ...])
                    If None, uses default MA cross grid.
        cost_config: Cost configuration
        initial_balance: Starting balance for each window
        size_lots: Position size
        symbol: Instrument name
    """
    cfg = cost_config or CostConfig()
    engine_kwargs = engine_kwargs or {}
    factory = strategy_factory or (lambda **params: MACrossStrategy(**params))

    # Default param grid for MA cross
    if param_grid is None:
        param_grid = []
        for fast in range(5, 51, 5):
            for slow in range(20, 201, 10):
                if slow > fast * 1.5:
                    param_grid.append({"fast_n": fast, "slow_n": slow})

    # Prepare data
    data = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(data["ts_utc"]):
        data["ts_utc"] = pd.to_datetime(data["ts_utc"])
    data = data.sort_values("ts_utc").reset_index(drop=True)

    start = data["ts_utc"].iloc[0]
    end = data["ts_utc"].iloc[-1]

    windows: List[WindowResult] = []
    window_idx = 0
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > end:
            break

        train_df = data[(data["ts_utc"] >= train_start) & (data["ts_utc"] < train_end)]
        test_df = data[(data["ts_utc"] >= test_start) & (data["ts_utc"] < test_end)]

        if len(train_df) < 50 or len(test_df) < 20:
            cursor += pd.Timedelta(days=step_days)
            continue

        # Optimize: find best params on training set
        best_sharpe = -999
        best_params = param_grid[0] if param_grid else {"fast_n": 10, "slow_n": 30}

        for params in param_grid:
            strat = factory(**params)
            engine = BacktestEngine(strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs)
            result = engine.run(train_df)
            if result.metrics.sharpe_ratio > best_sharpe and result.metrics.total_trades > 3:
                best_sharpe = result.metrics.sharpe_ratio
                best_params = params

        # Evaluate best params on test set
        best_strat = factory(**best_params)
        train_engine = BacktestEngine(best_strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs)
        train_result = train_engine.run(train_df)

        test_strat = factory(**best_params)
        test_engine = BacktestEngine(test_strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs)
        test_result = test_engine.run(test_df)

        train_sharpe = train_result.metrics.sharpe_ratio
        test_sharpe = test_result.metrics.sharpe_ratio

        if abs(train_sharpe) > 0.01:
            degradation = (train_sharpe - test_sharpe) / abs(train_sharpe)
        else:
            degradation = 0.0

        windows.append(WindowResult(
            window_idx=window_idx,
            train_start=str(train_start),
            train_end=str(train_end),
            test_start=str(test_start),
            test_end=str(test_end),
            best_params=best_params,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            test_profit_factor=test_result.metrics.profit_factor,
            test_total_trades=test_result.metrics.total_trades,
            degradation=degradation,
        ))

        window_idx += 1
        cursor += pd.Timedelta(days=step_days)

    # Aggregate
    total = len(windows)
    profitable = sum(1 for w in windows if w.test_profit_factor > 1.0)
    consistency = (profitable / total * 100) if total > 0 else 0.0
    avg_sharpe = float(np.mean([w.test_sharpe for w in windows])) if windows else 0.0
    avg_deg = float(np.mean([w.degradation for w in windows])) if windows else 0.0

    return WalkForwardResult(
        windows=windows,
        profitable_windows=profitable,
        total_windows=total,
        consistency_pct=consistency,
        avg_oos_sharpe=avg_sharpe,
        avg_degradation=avg_deg,
    )
