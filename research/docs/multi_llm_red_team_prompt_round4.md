# Multi-LLM Red-Team Prompt (Round 4 Final Verification)

## Mission
You are performing a final verification audit of a BTC perp backtesting/research repo after round-3 hardening plus a final tick-rounding precision fix.

Goal: confirm whether the system is now robust enough to be classified as `Paper-trade ready` or still `Research-only`.

## Hard Constraints
- BTC perp scope only.
- Real data only.
- No fake fills, no fabricated assumptions, no invented outputs.
- If runtime is unavailable, state it explicitly and continue with static audit.

## Items You Must Verify
1. Tick rounding precision regression is fixed and tests are green.
2. Loader fails closed on symbol/market mismatch and funding coverage gaps.
3. Slippage lookahead cannot be reintroduced via execution API defaults.
4. Correlation gate rejects inverse clones when absolute mode is enabled.
5. Scoring uses normalized/annualized turnover signal and does not silently accept unsafe metric fallbacks.
6. Final config remains stricter than screening config in practice.

## Priority Files
- `research/engine/execution.py`
- `research/data/loader.py`
- `research/engine/scoring.py`
- `research/engine/metrics.py`
- `research/cli.py`
- `research/config/backtest_screen.yml`
- `research/config/backtest_final.yml`
- `research/tests/test_execution_constraints.py`
- `research/tests/test_execution_costs.py`
- `research/tests/test_loader_guards.py`
- `research/tests/test_correlation_gate.py`
- `research/tests/test_scoring_turnover.py`

## Runtime Commands (if available)
```bash
PYTHONPATH=. pytest -q
PYTHONPATH=. python -m research.cli run --data-config research/config/data.yml --backtest-config research/config/backtest_screen.yml --candidates 1 --timeframes 1h
```

If CLI fails, classify as expected fail-closed vs regression.

## Required Output Format
1. Executive Verdict
2. Verification Matrix (6 items above: Correct / Partial / Not)
3. Severity-Ranked Findings (Critical/High/Medium/Low)
4. Remaining Promotion Blockers (max 5)
5. Top 10 Next Changes (if any)
6. Unknowns / Limitations
7. Risk Scores (correctness, execution realism, overfitting, operational, confidence)
8. Promotion Readiness with explicit conditions
9. Runtime Evidence

## Quality Bar
- Be concrete and adversarial.
- Tie each major claim to file-level evidence.
- Keep recommendations specific to this repo.
