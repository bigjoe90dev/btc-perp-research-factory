from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def write_daily_report(
    run_dir: Path,
    dataset_summary: dict[str, Any],
    outcomes: list[dict[str, Any]],
    assumptions: list[str],
    unknowns: list[str],
    top_overall: int,
    top_per_family: int,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "report.md"

    tested = len(outcomes)
    failed = sum(1 for x in outcomes if not x.get("gates", {}).get("passed_hard", False))
    survivors = [x for x in outcomes if x.get("gates", {}).get("passed_hard", False)]
    selected = [x for x in outcomes if x.get("selected", False)]
    corr_rejected = sum(
        1
        for x in outcomes
        if any(str(r).startswith("correlation>") for r in x.get("gates", {}).get("hard_reject_reasons", []))
    )

    ranked = sorted(outcomes, key=lambda x: float(x.get("score_adjusted", -1e9)), reverse=True)
    top_all = ranked[:top_overall]

    by_family = defaultdict(list)
    for row in ranked:
        by_family[str(row.get("family"))].append(row)

    lines: list[str] = []
    lines.append("# Daily Research Report")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append(f"- dataset_id: `{dataset_summary.get('dataset_id')}`")
    lines.append(f"- candles: `{dataset_summary.get('candles_path')}`")
    lines.append(f"- funding: `{dataset_summary.get('funding_path')}`")
    lines.append(f"- candles_start: `{dataset_summary.get('candles_start')}`")
    lines.append(f"- candles_end: `{dataset_summary.get('candles_end')}`")
    if dataset_summary.get("full_candles_start") is not None:
        lines.append(f"- full_candles_start: `{dataset_summary.get('full_candles_start')}`")
    if dataset_summary.get("full_candles_end") is not None:
        lines.append(f"- full_candles_end: `{dataset_summary.get('full_candles_end')}`")
    lines.append(f"- candle_rows: `{dataset_summary.get('candle_rows')}`")
    lines.append(f"- gaps_found: `{dataset_summary.get('gaps_found')}`")
    lines.append(f"- duplicates_found: `{dataset_summary.get('duplicates_found')}`")
    lines.append(f"- funding_start: `{dataset_summary.get('funding_start')}`")
    lines.append(f"- funding_end: `{dataset_summary.get('funding_end')}`")
    lines.append(f"- funding_rows: `{dataset_summary.get('funding_rows')}`")
    if dataset_summary.get("windowing_enabled"):
        lines.append(f"- optimization_window: `{dataset_summary.get('optimization_window_start')} -> {dataset_summary.get('optimization_window_end')}`")
        lines.append(f"- oos_window: `{dataset_summary.get('oos_window_start')} -> {dataset_summary.get('oos_window_end')}`")
        lines.append(f"- robustness_window: `{dataset_summary.get('robustness_window_start')} -> {dataset_summary.get('robustness_window_end')}`")
    lines.append("")

    lines.append("## Candidate Summary")
    lines.append(f"- generated: `{tested}`")
    lines.append(f"- failed_gates: `{failed}`")
    lines.append(f"- survivors: `{len(survivors)}`")
    lines.append(f"- selected_after_correlation: `{len(selected)}`")
    lines.append(f"- correlation_rejected: `{corr_rejected}`")
    lines.append("")

    lines.append("## Leaderboard Overall")
    lines.append("| strategy_id | family | timeframe | score_adj | return | sharpe | oos_ret | oos_sh | max_dd | skew | kurt | selected | gates |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for row in top_all:
        m = row.get("aggregate_metrics", {})
        oos = row.get("oos_metrics") or {}
        oos_ret = "n/a" if not oos else _fmt_pct(float(oos.get("total_return", 0.0)))
        oos_sh = "n/a" if not oos else f"{float(oos.get('sharpe', 0.0)):.2f}"
        lines.append(
            "| {sid} | {fam} | {tf} | {score:.4f} | {ret} | {sh:.2f} | {oos_ret} | {oos_sh} | {dd} | {sk:.2f} | {ku:.2f} | {sel} | {gate} |".format(
                sid=row.get("strategy_id"),
                fam=row.get("family"),
                tf=row.get("timeframe"),
                score=float(row.get("score_adjusted", 0.0)),
                ret=_fmt_pct(float(m.get("total_return", 0.0))),
                sh=float(m.get("sharpe", 0.0)),
                oos_ret=oos_ret,
                oos_sh=oos_sh,
                dd=_fmt_pct(float(m.get("max_drawdown", 0.0))),
                sk=float(m.get("return_skewness", 0.0)),
                ku=float(m.get("return_excess_kurtosis", 0.0)),
                sel="YES" if row.get("selected", False) else "NO",
                gate="PASS" if row.get("gates", {}).get("passed_hard", False) else "FAIL",
            )
        )
    lines.append("")

    lines.append("## Leaderboard By Family")
    for fam in sorted(by_family.keys()):
        lines.append(f"### {fam}")
        lines.append("| strategy_id | timeframe | score_adj | return | sharpe | gate |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in by_family[fam][:top_per_family]:
            m = row.get("aggregate_metrics", {})
            lines.append(
                "| {sid} | {tf} | {score:.4f} | {ret} | {sh:.2f} | {gate} |".format(
                    sid=row.get("strategy_id"),
                    tf=row.get("timeframe"),
                    score=float(row.get("score_adjusted", 0.0)),
                    ret=_fmt_pct(float(m.get("total_return", 0.0))),
                    sh=float(m.get("sharpe", 0.0)),
                    gate="PASS" if row.get("gates", {}).get("passed_hard", False) else "FAIL",
                )
            )
        lines.append("")

    lines.append("## Robustness")
    for row in top_all[: min(3, len(top_all))]:
        st = row.get("stress", {})
        lines.append(f"- `{row.get('strategy_id')}` cost_pass={st.get('cost_sweep', {}).get('pass')} "
                     f"latency_pass={st.get('latency_sweep', {}).get('pass')} "
                     f"perturb_pass={st.get('parameter_perturbation', {}).get('pass')} "
                     f"bootstrap_prob_negative={st.get('bootstrap', {}).get('prob_negative')}")
    if corr_rejected > 0:
        lines.append(f"- correlation gate rejected `{corr_rejected}` candidate(s).")
    lines.append("")

    lines.append("## Assumptions")
    for item in assumptions:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## DO NOT DEPLOY")
    if unknowns:
        for item in unknowns:
            lines.append(f"- {item}")
    else:
        lines.append("- No explicit unknowns flagged.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
