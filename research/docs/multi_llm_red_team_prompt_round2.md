# Multi-LLM Red-Team Prompt (Round 2 Verification)

## Mission
You are performing a **second-round red-team audit** of a BTC perp research/backtesting repo after a focused fix pack was applied.

Your objective is to verify whether the fixes are **actually implemented correctly**, identify regressions, and state what still blocks promotion.

## Hard Constraints
- BTC perp scope only.
- Real data only.
- No fabricated assumptions, no fake fills, no invented outputs.
- If runtime is impossible, say so clearly and continue with static audit.

## Fix Pack Claimed by Engineering (You Must Verify)
1. Slippage lookahead removed:
   - Slippage should use a decision-time bar, not future bar high/low.
2. Funding coverage now fail-closed:
   - Funding coverage enforcement is enabled in data config.
3. Turnover penalty normalization:
   - Scoring should use normalized turnover (ratio), not raw USD turnover.
4. `backtest_final.yml` CV windows upgraded:
   - train/test/step should no longer be tiny.
5. Stress + correlation now use OOS-only simulation path:
   - Not full-window IS+OOS leakage.
6. Correlation gate updated to absolute correlation:
   - inverse-signal clones should be catchable.

## Files You Must Inspect
- `research/engine/execution.py`
- `research/engine/simulator.py`
- `research/engine/metrics.py`
- `research/engine/scoring.py`
- `research/cli.py`
- `research/config/data.yml`
- `research/config/backtest_screen.yml`
- `research/config/backtest_final.yml`
- `research/tests/test_execution_costs.py`
- `research/tests/test_scoring_turnover.py`

## Mandatory Checks
1. Verify each of the 6 fix claims above with file-level evidence.
2. Search for side effects/regressions from these fixes.
3. Validate that scoring dimensions are now coherent.
4. Validate that OOS-only logic is not accidentally reintroducing leakage elsewhere.
5. Validate that stricter funding coverage does not silently degrade behavior.

## Runtime Checks (if execution available)
Run:

```bash
PYTHONPATH=. pytest -q research/tests/test_no_lookahead.py research/tests/test_purged_splits.py research/tests/test_execution_costs.py research/tests/test_funding_alignment.py research/tests/test_correlation_gate.py research/tests/test_scoring_turnover.py
PYTHONPATH=. python -m research.cli run --data-config research/config/data.yml --backtest-config research/config/backtest_screen.yml --candidates 1 --timeframes 1h
```

If CLI fails, determine whether failure is:
- expected fail-closed behavior, or
- a regression bug.

## Output Format (strict)
Use this exact section order:

1. Executive Verdict
2. Verification Matrix (6 claimed fixes)
   - For each: `Implemented Correctly | Partially | Not Implemented`
   - include file references and short evidence.
3. Severity-Ranked Findings (Critical/High/Medium/Low)
4. Regression Risks Introduced by Fix Pack
5. Top 10 Next Changes (ordered)
6. Unknowns / Limitations
7. Risk Scores
   - correctness, execution realism, overfitting risk, operational risk, audit confidence
8. Promotion Readiness (`Not ready` / `Research-only` / `Paper-trade ready`) + conditions
9. Runtime Evidence

## Review Quality Bar
- Be concrete and adversarial.
- Tie every major claim to code evidence.
- Do not give generic trading advice.
