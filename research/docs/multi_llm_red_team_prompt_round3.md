# Multi-LLM Red-Team Prompt (Round 3 Hardening Verification)

## Mission
You are performing a third-round verification audit on a BTC perpetual research/backtesting system after additional hardening fixes.

Your job is to confirm whether hardening changes are truly correct, identify any new regressions, and clearly state what still blocks paper-trade readiness.

## Hard Constraints
- Scope: BTC perp only.
- Real data only.
- No fabricated assumptions, no invented outputs.
- If runtime is unavailable, explicitly say so and continue static audit.

## Hardening Fixes Claimed in Round 3 (Verify All)
1. Loader now fail-closed on dataset scope mismatch (`required_symbol` and `required_market`).
2. Execution no longer allows slippage lookahead fallback:
   - `slippage_bar` is required by execution API.
3. Added funding coverage fail-closed regression test.
4. Added absolute-correlation regression test for inverse-signal clones.
5. Turnover penalty now uses annualized normalized turnover signal.
6. Final config (`backtest_final.yml`) is stricter than screen/default config on key gates and CV windows.

## Files You Must Inspect
- `research/data/loader.py`
- `research/engine/execution.py`
- `research/engine/metrics.py`
- `research/engine/scoring.py`
- `research/config/backtest_screen.yml`
- `research/config/backtest_final.yml`
- `research/config/backtest.yml`
- `research/tests/test_loader_guards.py`
- `research/tests/test_correlation_gate.py`
- `research/tests/test_execution_costs.py`
- `research/tests/test_scoring_turnover.py`
- `research/cli.py`

## Mandatory Verification Steps
1. Build a 6-row verification matrix for the claimed hardening fixes with:
   - `Implemented Correctly`, `Partially Implemented`, or `Not Implemented`.
2. Confirm final-vs-screen rigor ordering:
   - CV windows
   - trade count logic
   - robustness gates
   - OOS requirements
3. Confirm there is no obvious reintroduction path for execution lookahead.
4. Confirm loader rejects non-BTC/non-perp datasets before runtime.
5. Confirm turnover penalty dimensions are coherent and comparable across run lengths.

## Runtime Checks (if execution available)
Run:

```bash
PYTHONPATH=. pytest -q research/tests/test_no_lookahead.py research/tests/test_purged_splits.py research/tests/test_execution_costs.py research/tests/test_funding_alignment.py research/tests/test_correlation_gate.py research/tests/test_scoring_turnover.py research/tests/test_loader_guards.py
PYTHONPATH=. python -m research.cli run --data-config research/config/data.yml --backtest-config research/config/backtest_screen.yml --candidates 1 --timeframes 1h
```

If CLI fails, classify whether failure is expected fail-closed behavior or a regression.

## Output Format (strict)
1. Executive Verdict
2. Verification Matrix (6 hardening fixes)
3. Severity-Ranked Findings (Critical/High/Medium/Low)
4. Regression Risks Introduced by Hardening
5. Top 10 Next Changes
6. Unknowns / Limitations
7. Risk Scores
8. Promotion Readiness + explicit conditions
9. Runtime Evidence

## Quality Bar
- Be concrete and adversarial.
- Every major claim must cite file-level evidence.
- Do not give generic trading advice.
