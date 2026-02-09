"""
Monte Carlo robustness testing.

1. Parameter sensitivity — sample N parameter sets, run backtest for each,
   compute what % produce acceptable results.
2. Bootstrap resampling — resample trade returns to build confidence intervals.
3. Equity path simulation — shuffle trade order to see range of possible outcomes.
"""

import random as _random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, MACrossStrategy, Strategy
from backtest.costs import CostConfig


@dataclass
class ParamSensitivityResult:
    n_samples: int
    n_robust: int
    robustness_pct: float               # % of param sets with sharpe > threshold
    sharpe_distribution: List[float]     # all sharpes
    best_params: dict
    best_sharpe: float
    worst_params: dict
    worst_sharpe: float
    median_sharpe: float

    def summary(self) -> str:
        return (
            f"=== Monte Carlo Parameter Sensitivity ===\n"
            f"  Samples:    {self.n_samples}\n"
            f"  Robust:     {self.n_robust} ({self.robustness_pct:.1f}%)\n"
            f"  Median Sharpe: {self.median_sharpe:.2f}\n"
            f"  Best:  {self.best_sharpe:.2f} {self.best_params}\n"
            f"  Worst: {self.worst_sharpe:.2f} {self.worst_params}\n"
        )


@dataclass
class BootstrapResult:
    n_iterations: int
    sharpe_p5: float
    sharpe_p50: float
    sharpe_p95: float
    max_dd_p5: float
    max_dd_p50: float
    max_dd_p95: float
    cagr_p5: float
    cagr_p50: float
    cagr_p95: float

    def summary(self) -> str:
        return (
            f"=== Bootstrap Confidence Intervals ({self.n_iterations} iterations) ===\n"
            f"  Sharpe:  p5={self.sharpe_p5:.2f}  p50={self.sharpe_p50:.2f}  p95={self.sharpe_p95:.2f}\n"
            f"  Max DD:  p5={self.max_dd_p5:.2%}  p50={self.max_dd_p50:.2%}  p95={self.max_dd_p95:.2%}\n"
            f"  CAGR:    p5={self.cagr_p5:.2%}  p50={self.cagr_p50:.2%}  p95={self.cagr_p95:.2%}\n"
        )


@dataclass
class EquityPathResult:
    n_paths: int
    terminal_wealth: np.ndarray          # array of final equity values
    median_terminal: float
    p5_terminal: float
    p95_terminal: float
    paths: Optional[np.ndarray] = None   # (n_paths, n_trades) equity paths

    def summary(self) -> str:
        return (
            f"=== Monte Carlo Equity Paths ({self.n_paths} paths) ===\n"
            f"  Terminal wealth:  p5=${self.p5_terminal:,.0f}  "
            f"p50=${self.median_terminal:,.0f}  p95=${self.p95_terminal:,.0f}\n"
        )


def parameter_sensitivity(
    df: pd.DataFrame,
    n_samples: int = 200,
    sharpe_threshold: float = 0.5,
    cost_config: CostConfig = None,
    initial_balance: float = 10_000.0,
    size_lots: float = 0.1,
    symbol: str = "EURUSD",
    seed: int = 42,
    strategy_factory=None,
    param_sampler=None,
    engine_kwargs: Optional[dict] = None,
) -> ParamSensitivityResult:
    """
    Sample random parameter sets, run backtests, measure robustness.

    Args:
        df: Candle data (ts_utc + ohlc)
        strategy_factory: callable(**params)->Strategy (default MACrossStrategy)
        param_sampler: callable(rng)->dict (default MA ranges)
        engine_kwargs: extra BacktestEngine args (signal_timing, spread_model, etc)
    """
    cfg = cost_config or CostConfig()
    rng = _random.Random(seed)
    engine_kwargs = engine_kwargs or {}

    def _default_sampler(rng_obj):
        fast_n = rng_obj.randint(5, 50)
        slow_n = rng_obj.randint(max(fast_n + 5, 20), 200)
        return {"fast_n": fast_n, "slow_n": slow_n}

    def _default_factory(**params):
        return MACrossStrategy(**params)

    sampler = param_sampler or _default_sampler
    factory = strategy_factory or _default_factory

    sharpes = []
    param_list = []

    for _ in range(n_samples):
        params = sampler(rng)
        strat: Strategy = factory(**params)
        engine = BacktestEngine(
            strat, cfg, initial_balance, size_lots, symbol, **engine_kwargs
        )
        result = engine.run(df)

        sharpes.append(result.metrics.sharpe_ratio)
        param_list.append(params)

    sharpes_arr = np.array(sharpes)
    robust = int(np.sum(sharpes_arr > sharpe_threshold))
    best_idx = int(np.argmax(sharpes_arr))
    worst_idx = int(np.argmin(sharpes_arr))

    return ParamSensitivityResult(
        n_samples=n_samples,
        n_robust=robust,
        robustness_pct=(robust / n_samples * 100) if n_samples > 0 else 0,
        sharpe_distribution=sharpes,
        best_params=param_list[best_idx],
        best_sharpe=sharpes[best_idx],
        worst_params=param_list[worst_idx],
        worst_sharpe=sharpes[worst_idx],
        median_sharpe=float(np.median(sharpes_arr)),
    )


def bootstrap_resampling(
    trade_pnls: List[float],
    n_iterations: int = 10_000,
    initial_balance: float = 10_000.0,
    bars_per_year: float = 252.0,
    seed: int = 42,
) -> BootstrapResult:
    """
    Resample trade returns with replacement to build confidence intervals.

    Args:
        trade_pnls: List of net P&L per trade
        n_iterations: Number of bootstrap iterations
        initial_balance: Starting balance
        bars_per_year: For Sharpe annualization (trades per year approximation)
    """
    rng = np.random.RandomState(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    if n_trades < 5:
        return BootstrapResult(
            n_iterations=n_iterations,
            sharpe_p5=0, sharpe_p50=0, sharpe_p95=0,
            max_dd_p5=0, max_dd_p50=0, max_dd_p95=0,
            cagr_p5=0, cagr_p50=0, cagr_p95=0,
        )

    sharpes = []
    max_dds = []
    cagrs = []

    ann = np.sqrt(bars_per_year)

    for _ in range(n_iterations):
        sample = rng.choice(pnls, size=n_trades, replace=True)

        # Build equity
        equity = np.cumsum(sample) + initial_balance

        # Sharpe from trade returns
        mean_r = np.mean(sample)
        std_r = np.std(sample, ddof=1)
        sharpe = (mean_r / std_r * ann) if std_r > 0 else 0
        sharpes.append(sharpe)

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / np.where(running_max > 0, running_max, 1)
        max_dds.append(abs(float(np.min(dd))))

        # CAGR approximation
        years = n_trades / bars_per_year
        final = equity[-1]
        if years > 0 and final > 0 and initial_balance > 0:
            cagr = (final / initial_balance) ** (1 / years) - 1
        else:
            cagr = 0
        cagrs.append(cagr)

    return BootstrapResult(
        n_iterations=n_iterations,
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p50=float(np.percentile(sharpes, 50)),
        sharpe_p95=float(np.percentile(sharpes, 95)),
        max_dd_p5=float(np.percentile(max_dds, 5)),
        max_dd_p50=float(np.percentile(max_dds, 50)),
        max_dd_p95=float(np.percentile(max_dds, 95)),
        cagr_p5=float(np.percentile(cagrs, 5)),
        cagr_p50=float(np.percentile(cagrs, 50)),
        cagr_p95=float(np.percentile(cagrs, 95)),
    )


def equity_path_simulation(
    trade_pnls: List[float],
    n_paths: int = 1_000,
    initial_balance: float = 10_000.0,
    seed: int = 42,
    store_paths: bool = False,
) -> EquityPathResult:
    """
    Shuffle trade order to simulate range of possible equity paths.
    Same trades, different sequence = different drawdown/equity experience.
    """
    rng = np.random.RandomState(seed)
    pnls = np.array(trade_pnls)
    n_trades = len(pnls)

    if n_trades < 3:
        return EquityPathResult(
            n_paths=0, terminal_wealth=np.array([initial_balance]),
            median_terminal=initial_balance, p5_terminal=initial_balance,
            p95_terminal=initial_balance,
        )

    terminals = np.empty(n_paths)
    paths_arr = np.empty((n_paths, n_trades)) if store_paths else None

    for i in range(n_paths):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled) + initial_balance
        terminals[i] = equity[-1]
        if store_paths:
            paths_arr[i] = equity

    return EquityPathResult(
        n_paths=n_paths,
        terminal_wealth=terminals,
        median_terminal=float(np.median(terminals)),
        p5_terminal=float(np.percentile(terminals, 5)),
        p95_terminal=float(np.percentile(terminals, 95)),
        paths=paths_arr,
    )
