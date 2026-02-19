from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Bar:
    ts_utc: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class OrderIntent:
    ts_utc: pd.Timestamp
    target_side: int  # -1 short, 0 flat, +1 long
    reason: str


@dataclass(frozen=True)
class Fill:
    ts_utc: pd.Timestamp
    side: int
    qty: float
    price: float
    fee_paid: float
    slippage_bps: float
    reason: str


@dataclass(frozen=True)
class FundingEvent:
    ts_utc: pd.Timestamp
    rate: float
    cashflow: float


@dataclass(frozen=True)
class Trade:
    entry_ts_utc: pd.Timestamp
    exit_ts_utc: pd.Timestamp
    side: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_after_costs: float
    bars_held: int


@dataclass
class SimulationResult:
    equity_curve: pd.DataFrame
    trades: list[Trade]
    fills: list[Fill]
    funding_events: list[FundingEvent]
    summary: dict[str, Any]


@dataclass(frozen=True)
class CandidateSpec:
    strategy_id: str
    family: str
    timeframe: str
    params: dict[str, Any]
    rules_version: str
    dataset_key: str


@dataclass
class CandidateOutcome:
    spec: CandidateSpec
    fold_metrics: list[dict[str, Any]]
    aggregate_metrics: dict[str, Any]
    stress: dict[str, Any]
    gates: dict[str, Any]
    score_raw: float
    score_adjusted: float
    selected: bool = False
    notes: list[str] = field(default_factory=list)
