# Repo Red-Team Context (BTC Perp Research Stack)

## Scope
- In scope: `research/` BTC perp data + backtest/research stack.
- Out of scope: non-BTC strategy development and unrelated legacy components.

## Key Paths
- CLI entrypoint: `research/cli.py`
- Data loading/integrity/funding:
  - `research/data/loader.py`
  - `research/data/integrity.py`
  - `research/data/funding.py`
- Engine:
  - `research/engine/simulator.py`
  - `research/engine/execution.py`
  - `research/engine/cv.py`
  - `research/engine/gates.py`
  - `research/engine/stress.py`
  - `research/engine/metrics.py`
  - `research/engine/scoring.py`
- Strategies:
  - `research/strategies/base.py`
  - `research/strategies/registry.py`
  - `research/strategies/generator.py`
  - `research/strategies/families/*.py`
- Reports:
  - `research/reports/daily.py`
  - `research/reports/checklist.py`
- Tests:
  - `research/tests/test_no_lookahead.py`
  - `research/tests/test_purged_splits.py`
  - `research/tests/test_execution_costs.py`
  - `research/tests/test_funding_alignment.py`
  - `research/tests/test_correlation_gate.py`
  - and related tests in `research/tests/`

## Data and Config Assumptions
- Data config: `research/config/data.yml`
  - primary dataset key: `BTC_BITMEX_PERP_1M`
- Backtest configs:
  - screening: `research/config/backtest_screen.yml`
  - stricter: `research/config/backtest_final.yml`
- Manifest: `research/data_manifest.yml`

## Verified Baseline Facts
1. Generator uses deterministic strategy IDs and deterministic ordering.
2. Partial-count selection uses round-robin bucket selection by `(family, timeframe)`.
3. Execution constraints include participation cap, min notional, qty step, tick rounding.
4. Correlation gate is applied post-ranking to reduce clone survivors.
5. No-lookahead API blocks explicit future access in strategy context.

## Known Strengths
- Explicit anti-lookahead interfaces.
- Walk-forward + purge + embargo support.
- Funding integration in PnL flow.
- Cost/latency/parameter perturbation stress scaffolding.
- Correlation filtering on selected candidates.

## Known Gaps / Risks To Validate
- Whether all gates are too strict or too lenient for actual BTC perp behavior.
- Whether one-candidate sanity run meaningfully validates end-to-end behavior.
- Whether strategy family parameter pools are sufficiently diverse vs. regime-specific overfitting.
- Whether report outputs can hide fragility if user reads only summary metrics.

## Exact Commands For Fast Validation
```bash
pytest -q research/tests/test_no_lookahead.py research/tests/test_purged_splits.py research/tests/test_execution_costs.py research/tests/test_funding_alignment.py research/tests/test_correlation_gate.py
PYTHONPATH=. .venv/bin/python -m research.cli run --data-config research/config/data.yml --backtest-config research/config/backtest_screen.yml --candidates 1 --timeframes 1h
```

## Expected Artifacts After 1-Candidate Run
- Latest folder under `research/artefacts/runs/<run_id>/`
- Required files:
  - `report.md`
  - `gates.json`
  - `folds.json`
  - `results.parquet`

## Review Output Requirement
Use `research/docs/llm_review_response_template.md` so outputs from different LLMs can be compared directly.
