# Strategy Quality Checklist (No Fake Data)

Use this checklist to judge whether a strategy is worth promoting.
All checks are based on real historical candles/funding and realistic execution costs.

## Checklist
1. Enough activity
- It trades enough times to be meaningful in the tested window.

2. Fold consistency
- It performs acceptably across many time slices, not only one lucky patch.

3. Positive net outcome
- It is positive after all simulated costs.

4. Risk-adjusted quality
- Sharpe/quality is not deeply negative.

5. Drawdown control
- Largest loss period stays inside risk limits.

6. Cost robustness
- Still acceptable when costs are increased.

7. Latency robustness
- Still acceptable when fills are delayed.

8. Parameter stability
- Small parameter tweaks do not collapse performance.

9. Bootstrap fragility
- Resampled outcomes do not show high probability of failure.

10. Tail behavior
- Avoids extreme blow-up style return shapes.

11. Win/loss quality
- Average win/loss profile is not structurally poor.

12. Out-of-sample sanity
- If OOS is evaluated, it should not be clearly negative.

## How to use
- Screening stage: use checklist score to shortlist ideas.
- Final stage: require hard gate pass plus strong checklist score.
- Always inspect trade behavior and assumptions before promotion.

## Generated artifacts per run
- `checklist_scores.parquet`
- `checklist_scores.csv`
- `checklist_summary.md`
