# Multi-LLM Red-Team Prompt: BTC Perp Research Stack

## How To Use This Prompt
- Send this file and the repository zip to each LLM.
- Ask each LLM to follow every section exactly.
- Do not let the model skip runtime checks unless it explicitly cannot run code.

## Mission
You are acting as an independent red-team auditor for a BTC perpetual strategy research/backtesting system.

Your job is to find failure modes that could create false confidence and misleading profitability claims.

## Non-Negotiable Constraints
- Instrument scope: BTC perp only.
- Data policy: real data only; no synthetic data.
- Truth policy: no fake fills, no fake results, no fabricated assumptions.
- Research policy: fail-closed behavior is expected when data/coverage/integrity is insufficient.

## Repo Context You Must Assume
1. Candidate generator includes deterministic round-robin selection across `(family, timeframe)` buckets.
2. 288 requested candidates should span 5 families and 2 timeframes (`5m`, `1h`) with equal timeframe split.
3. Execution model includes volume participation cap, min notional, qty step rounding, tick rounding, slippage, and delay.
4. Correlation gate exists and has tests.
5. Runtime validation should include one fast candidate run (`--candidates 1 --timeframes 1h`).

## What You Must Audit
1. Data integrity and fail-closed behavior.
2. No-lookahead enforcement in strategy API and simulator flow.
3. CV leakage control (walk-forward, purging, embargo, windowing).
4. Execution realism:
   - fees
   - slippage
   - latency/delay
   - participation caps
   - min_notional / tick / lot effects
5. Funding alignment and funding PnL correctness.
6. Multiple-testing controls and ranking logic.
7. Gate strictness/logic, including correlation gate.
8. Candidate generation diversity and risk of clone strategies.
9. Reporting correctness and whether reports can mislead.

## Mandatory Runtime Check (if code execution is possible)
Run exactly:

```bash
pytest -q research/tests/test_no_lookahead.py research/tests/test_purged_splits.py research/tests/test_execution_costs.py research/tests/test_funding_alignment.py research/tests/test_correlation_gate.py
PYTHONPATH=. .venv/bin/python -m research.cli run --data-config research/config/data.yml --backtest-config research/config/backtest_screen.yml --candidates 1 --timeframes 1h
```

Then verify latest run folder contains:
- `report.md`
- `gates.json`
- `folds.json`
- `results.parquet`

If execution is not possible, clearly state that and continue with static audit only.

## Required Output Format
Use this exact section structure:

1. **Executive Verdict**
2. **Severity-Ranked Findings**
   - Critical
   - High
   - Medium
   - Low
   - Each finding must include file path(s) and why it matters.
3. **What Is Strong**
4. **What Can Produce False Confidence**
5. **Top 15 Prioritized Changes**
   - include impact and implementation difficulty.
6. **Strategy Quality Guidance (Anti-Overfitting)**
   - concrete recommendations only.
7. **Unknowns / Limitations**
   - explicitly state what a 1-candidate run cannot prove.
8. **Risk Scores (0-10, higher is worse)**
   - Correctness risk
   - Execution realism risk
   - Overfitting/data-snooping risk
   - Operational risk
   - Audit confidence (inverse: higher is better confidence)
9. **Promotion Readiness**
   - `Not ready`, `Research-only`, or `Paper-trade ready`
   - with explicit conditions to move to the next level.

## Strict Review Rules
- Do not provide generic advice without tying it to repo code/config.
- Do not claim profitable strategy design unless justified by evidence from code/results.
- Do not optimize for headline Sharpe only; penalize fragility and hidden tail risk.
- Do not ignore exchange microstructure assumptions.
- If any conclusion is uncertain, label it as uncertain and explain what evidence is missing.

## Deliverable Quality Bar
Your output should be decision-useful for an engineering team to execute immediately.
No fluff, no motivational language, no hand-waving.
