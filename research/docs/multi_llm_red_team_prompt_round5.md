# Round-5 Full Repo Red-Team Audit Prompt (Generator-Focused)

You are acting as an independent red-team auditor for a BTC-perp backtesting/research repository.

## Inputs
- Repo link: `<PASTE_GITHUB_LINK>`
- Optional zip: `<PASTE_ZIP_LINK_OR_ATTACH_ZIP>`

## Mission
Audit the entire repository for correctness, realism, leakage risk, operational safety, and false-confidence risk.
Prioritize the new generator system and its integration path into the existing backtester.

## Non-Negotiable Constraints
- BTC perp only.
- Real data only.
- No fabricated results, no invented assumptions.
- If runtime is unavailable, explicitly state static-only limits.
- Every finding must cite concrete file paths.

## Priority Weighting
- 60%: generator system and candidate plumbing
- 25%: scoring/gates/CV/anti-leakage interactions
- 15%: operational workflow and reproducibility

## Critical Areas You Must Verify

### A) Generator System (highest priority)
Review these files deeply:
- `research/generator/config.py`
- `research/generator/openrouter_client.py`
- `research/generator/data_summary.py`
- `research/generator/schema.py`
- `research/generator/adapter.py`
- `research/generator/prefilter.py`
- `research/generator/prompting.py`
- `research/generator/pipeline.py`
- `research/generator/prompts/master_btc_perp_generator_prompt_v1.md`

Verify:
1. Two-round flow exists (generation + synthesis), and logic is coherent.
2. Fail-closed behavior if required Grok lane/synthesis fails.
3. Budget limits are enforced (calls/tokens).
4. Schema validation rejects malformed or out-of-scope candidates.
5. Adapter restricts to existing strategy families and safe param ranges.
6. Diversity/dedup logic is effective and deterministic.
7. Artifacts are complete and reproducible:
   - `generation_manifest.json`
   - `raw_model_outputs.jsonl`
   - `candidates_raw.json`
   - `candidates_validated.json`
   - `rejections.json`
   - `generation_report.md`
8. Prompt version/hash/rendered prompt are persisted for reproducibility.
9. Data summary is rich and grounded in real dataset stats.
10. No unsafe dynamic code execution path exists.

### B) Integration Into Backtester
Review:
- `research/cli.py`
- `research/strategies/candidate_ids.py`
- `research/strategies/generator.py`
- `research/engine/scoring.py`
- `research/config/backtest.yml`
- `research/config/backtest_screen.yml`
- `research/config/backtest_final.yml`
- `research/config/data.yml`

Verify:
1. `run --candidate-file` integration is correct and safe.
2. New `generate_candidates` and `doctor_data` commands work as intended.
3. Scoring fail-closed on missing `turnover_ratio_annualized` is correct.
4. Funding tolerance defaults are consistent.
5. Legacy deterministic generation path still works unchanged.

## Mandatory Runtime Checks (if environment allows)
Run exactly:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q research/tests/test_scoring_turnover.py research/tests/test_execution_constraints.py research/tests/test_loader_guards.py research/tests/test_correlation_gate.py research/tests/test_execution_costs.py research/tests/test_determinism.py research/tests/test_llm_adapter.py research/tests/test_llm_prefilter.py research/tests/test_cli_candidate_file.py research/tests/test_llm_pipeline_estimate_only.py

PYTHONPATH=. .venv/bin/python -m research.cli doctor_data --data-config research/config/data.yml

PYTHONPATH=. .venv/bin/python -m research.cli generate_candidates --data-config research/config/data.yml --backtest-config research/config/backtest.yml --count 5 --timeframes 1h --estimate-only
```

If OpenRouter key + network are available, also run:

```bash
PYTHONPATH=. .venv/bin/python -m research.cli generate_candidates --data-config research/config/data.yml --backtest-config research/config/backtest.yml --count 10 --timeframes 1h
```

If this cannot be run, clearly state why.

## Required Output Format
Use this exact section order:

1. Executive Verdict
2. Verification Matrix (Generator + Integration claims)
3. Severity-Ranked Findings
   - Critical
   - High
   - Medium
   - Low
4. What Is Strong
5. What Can Produce False Confidence
6. Top 15 Prioritized Changes (impact + difficulty)
7. Unknowns / Limitations
8. Risk Scores (0-10)
   - Correctness risk
   - Execution realism risk
   - Overfitting/data-snooping risk
   - Operational risk
   - Audit confidence (higher is better)
9. Promotion Readiness
   - `Not ready`, `Research-only`, or `Paper-trade ready`
   - explicit conditions
10. Runtime Check Evidence
   - tests run
   - commands run
   - artifacts found/not found
   - exact blockers

## Strict Review Rules
- No fluff.
- No generic advice without file-level evidence.
- No profitability claims.
- Call out any silent fallback that can corrupt results.
- Highlight any place where missing keys/data could silently pass.

