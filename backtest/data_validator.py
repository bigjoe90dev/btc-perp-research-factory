"""
Data integrity validation for backtest input data.
Checks: missing bars, outliers, duplicates, bad spreads, coverage.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationReport:
    total_bars: int = 0
    expected_bars: int = 0
    coverage_pct: float = 0.0
    gap_count: int = 0
    outlier_count: int = 0
    duplicate_count: int = 0
    bad_spread_count: int = 0
    issues: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len([i for i in self.issues if i.startswith("CRITICAL")]) == 0

    def summary(self) -> str:
        status = "PASS" if self.is_clean else "FAIL"
        lines = [
            f"Data Validation: {status}",
            f"  Bars: {self.total_bars} / {self.expected_bars} expected ({self.coverage_pct:.1f}%)",
            f"  Gaps: {self.gap_count}",
            f"  Outliers: {self.outlier_count}",
            f"  Duplicates: {self.duplicate_count}",
            f"  Bad spreads: {self.bad_spread_count}",
        ]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        return "\n".join(lines)


def validate_quotes(df: pd.DataFrame,
                    expected_interval_sec: float = 1.0,
                    outlier_threshold: float = 0.005,
                    min_coverage: float = 0.50) -> ValidationReport:
    """
    Validate a DataFrame of quote data before running a backtest.

    Args:
        df: DataFrame with columns: ts_utc, bid, ask (ts_utc as datetime)
        expected_interval_sec: Expected interval between bars in seconds
        outlier_threshold: Max expected single-bar return (0.005 = 0.5%)
        min_coverage: Minimum required coverage ratio (0.50 = 50%)

    Returns:
        ValidationReport with all issues found
    """
    report = ValidationReport()

    if df.empty:
        report.issues.append("CRITICAL: DataFrame is empty")
        return report

    # Ensure ts_utc is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df = df.copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"])

    df = df.sort_values("ts_utc").reset_index(drop=True)
    report.total_bars = len(df)

    # 1. Coverage check
    time_span = (df["ts_utc"].iloc[-1] - df["ts_utc"].iloc[0]).total_seconds()
    report.expected_bars = max(int(time_span / expected_interval_sec), 1)
    report.coverage_pct = (report.total_bars / report.expected_bars) * 100

    if report.coverage_pct < min_coverage * 100:
        report.issues.append(
            f"CRITICAL: Only {report.coverage_pct:.1f}% bar coverage "
            f"(need >{min_coverage*100:.0f}%)"
        )

    # 2. Gap detection (gaps > 10x expected interval)
    time_diffs = df["ts_utc"].diff().dt.total_seconds()
    gap_threshold = expected_interval_sec * 10
    gaps = time_diffs[time_diffs > gap_threshold]
    report.gap_count = len(gaps)
    if report.gap_count > 0:
        max_gap = gaps.max()
        report.issues.append(
            f"WARNING: {report.gap_count} gaps > {gap_threshold:.0f}s "
            f"(largest: {max_gap:.0f}s)"
        )

    # 3. Outlier detection (extreme price moves)
    mid = (df["bid"] + df["ask"]) / 2
    returns = mid.pct_change().abs()
    outliers = returns[returns > outlier_threshold]
    report.outlier_count = len(outliers)
    if report.outlier_count > 0:
        worst = outliers.max()
        report.issues.append(
            f"WARNING: {report.outlier_count} extreme price moves "
            f"(>{outlier_threshold*100:.1f}% per tick, worst: {worst*100:.2f}%)"
        )

    # 4. Bad spread detection (zero, negative, or inverted)
    spread = df["ask"] - df["bid"]
    bad_spread = spread[spread <= 0]
    report.bad_spread_count = len(bad_spread)
    if report.bad_spread_count > 0:
        report.issues.append(
            f"CRITICAL: {report.bad_spread_count} bars with zero/negative spread"
        )

    # 5. Duplicate timestamp detection
    dupes = df.duplicated(subset=["ts_utc"], keep="first")
    report.duplicate_count = int(dupes.sum())
    if report.duplicate_count > 0:
        report.issues.append(
            f"WARNING: {report.duplicate_count} duplicate timestamps"
        )

    return report


def clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning: deduplicate, sort, remove zero/negative spreads.
    Returns a cleaned copy.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(out["ts_utc"]):
        out["ts_utc"] = pd.to_datetime(out["ts_utc"])

    # Sort by time
    out = out.sort_values("ts_utc").reset_index(drop=True)

    # Remove duplicates (keep first)
    out = out.drop_duplicates(subset=["ts_utc"], keep="first")

    # Remove bad spreads
    spread = out["ask"] - out["bid"]
    out = out[spread > 0].reset_index(drop=True)

    return out


def validate_candles(df: pd.DataFrame,
                     expected_minutes: int = 5,
                     max_spread_pips: float = 10.0) -> ValidationReport:
    """
    Validate candle data generated from ticks.
    """
    report = ValidationReport()
    if df.empty:
        report.issues.append("CRITICAL: Candle DataFrame is empty")
        return report

    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out["ts_utc"]):
        out["ts_utc"] = pd.to_datetime(out["ts_utc"], errors="coerce", utc=True)
        out = out.dropna(subset=["ts_utc"])
    out = out.sort_values("ts_utc").reset_index(drop=True)

    report.total_bars = len(out)
    if report.total_bars < 2:
        report.issues.append("CRITICAL: Not enough candle bars")
        return report

    expected_interval_sec = expected_minutes * 60
    time_span = (out["ts_utc"].iloc[-1] - out["ts_utc"].iloc[0]).total_seconds()
    report.expected_bars = max(int(time_span / expected_interval_sec), 1)
    report.coverage_pct = (report.total_bars / report.expected_bars) * 100

    if report.coverage_pct < 50:
        report.issues.append(f"CRITICAL: Only {report.coverage_pct:.1f}% bar coverage")

    # Gaps
    time_diffs = out["ts_utc"].diff().dt.total_seconds()
    gap_threshold = expected_interval_sec * 5
    gaps = time_diffs[time_diffs > gap_threshold]
    report.gap_count = len(gaps)
    if report.gap_count > 0:
        report.issues.append(f"WARNING: {report.gap_count} gaps > {gap_threshold:.0f}s")

    # Spread checks
    spread = out.get("spread_c_pips")
    if spread is not None:
        bad_spread = spread[spread <= 0]
        report.bad_spread_count = len(bad_spread)
        if report.bad_spread_count > 0:
            report.issues.append(f"CRITICAL: {report.bad_spread_count} bars with zero/negative spread")
        outliers = spread[spread > max_spread_pips]
        report.outlier_count = len(outliers)
        if report.outlier_count > 0:
            report.issues.append(f"WARNING: {report.outlier_count} bars with spread > {max_spread_pips} pips")

    # Duplicate timestamps
    dupes = out.duplicated(subset=["ts_utc"], keep="first")
    report.duplicate_count = int(dupes.sum())
    if report.duplicate_count > 0:
        report.issues.append(f"WARNING: {report.duplicate_count} duplicate timestamps")

    return report
