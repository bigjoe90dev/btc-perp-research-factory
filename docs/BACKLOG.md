# ICT-Bot — JIRA-Style Backlog
## Epics, Stories & Tasks

**Last Updated:** 2026-02-06

---

## Legend

| Priority | Meaning |
|----------|---------|
| P0 | Blocker — must fix before anything else |
| P1 | Critical — required for backtest engine |
| P2 | High — required for robustness/trust |
| P3 | Medium — important for production readiness |
| P4 | Low — nice-to-have / post-MVP |

| Status | Meaning |
|--------|---------|
| TODO | Not started |
| IN PROGRESS | Currently being worked on |
| DONE | Completed and verified |
| BLOCKED | Waiting on dependency |

---

## PHASE 0: STABILIZE (Foundation)
### Epic: ICT-0 — Stabilize Existing Codebase

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-001 | Bug | Fix log_event.py broken import | P0 | TODO | References `get_conn()` which doesn't exist in `shared/db.py`. Also references `events` table instead of `event_log`. Fix import to use `connect()` and correct table name. |
| ICT-002 | Task | Consolidate DB schema in init_db.py | P0 | TODO | `init_db.py` only creates `heartbeat` and `event_log` tables. Missing: `quotes`, `signals`, `paper_positions`, `paper_trades`. These are created ad-hoc in daemon code. Consolidate all table creation into one schema file. |
| ICT-003 | Task | Add new backtest tables to schema | P0 | TODO | Add tables: `backtest_runs`, `backtest_results`, `backtest_trades`, `news_events`, `regime_labels`. |
| ICT-004 | Task | Validate all daemons can start cleanly | P1 | TODO | Start each daemon (`heartbeat_writer`, `market_data_daemon`, `paper_engine_daemon`) and verify they boot without errors. Fix any import/config issues. |
| ICT-005 | Task | Create shared logger module | P1 | TODO | Replace ad-hoc `log_event.py` with proper `shared/logger.py` that writes to `event_log` table and optionally to console/file. |
| ICT-006 | Task | Update requirements.txt | P1 | TODO | Add missing deps: `numpy`, `scipy` (for backtest/Monte Carlo). Pin `ctrader-open-api`. Add `twisted` explicitly. |
| ICT-007 | Task | Write README.md with setup instructions | P4 | TODO | Document: prerequisites, .env setup, DB init, daemon startup, dashboard access. |

**Acceptance:** All daemons start without errors. DB schema is consolidated. No broken imports.

---

## PHASE 1: DATA PIPELINE (Data Foundation)
### Epic: ICT-1 — Robust Data Pipeline

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-010 | Story | Create data integrity validator | P1 | TODO | Module `backtest/data_validator.py` that checks: missing bars, outlier detection (extreme price moves), zero-volume/zero-spread bars, timestamp gaps, and duplicate detection. |
| ICT-011 | Task | Missing bar detection | P1 | TODO | Given a timeframe (e.g., 1s), detect gaps where expected bars are missing. Report % coverage. Flag if >5% missing. |
| ICT-012 | Task | Outlier detection | P1 | TODO | Flag bars where returns exceed configurable threshold (e.g., >0.5% in 1 tick for EURUSD = flash crash or bad tick). |
| ICT-013 | Task | Duplicate tick detection | P2 | TODO | Detect and flag/deduplicate rows with identical timestamps. |
| ICT-014 | Story | Historical data loader | P1 | TODO | Create utility to import historical OHLC data from CSV files into the `quotes` table for backtesting. Support multiple timeframes. |
| ICT-015 | Task | CSV import script | P1 | TODO | `scripts/import_historical_csv.py` — reads CSV (ts, open, high, low, close, volume or ts, bid, ask) and inserts into quotes table. |
| ICT-016 | Task | Data export utility | P3 | TODO | Export quotes/trades/signals from SQLite to CSV for external analysis. |
| ICT-017 | Task | Quote table indexing | P2 | TODO | Add indexes on `quotes(env, account_id, symbol_id, ts_utc)` for query performance during backtests over large datasets. |

**Acceptance:** Can validate data integrity before running backtests. Can import historical CSV data. Missing bars and outliers are detected and reported.

---

## PHASE 2: BACKTEST ENGINE (Core)
### Epic: ICT-2 — Core Backtest Engine with Realistic Costs

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-020 | Story | Core backtest runner | P0 | TODO | `backtest/engine.py` — offline engine that replays quote data, applies strategy, tracks positions, and computes results. Decoupled from live daemons. Operates on DataFrames, not live DB polling. |
| ICT-021 | Task | Strategy interface (abstract base) | P1 | TODO | Define `Strategy` ABC with `on_tick(bar) -> Signal` method. MA crossover as first implementation. Allows plugging in ICT methodology later. |
| ICT-022 | Story | Commission modeling | P0 | TODO | `backtest/costs.py` — model forex transaction costs: spread cost (bid-ask at entry/exit), per-lot commission (configurable, default $3.50/lot/side), round-trip cost calculation. |
| ICT-023 | Task | Spread cost at entry/exit | P0 | TODO | Capture actual bid-ask spread at moment of trade. Apply half-spread cost to entry and exit. |
| ICT-024 | Story | Slippage simulation | P0 | TODO | Model slippage as: fixed component (configurable pips) + variable component based on recent volatility. Default: 0.2 pips fixed + 0.1x ATR for EURUSD. |
| ICT-025 | Story | Swap rate costs | P1 | TODO | Model overnight financing: if position held past daily rollover (typically 17:00 EST / 22:00 UTC), apply swap rate. Configurable per-symbol swap (long/short different rates). Forex equivalent of crypto funding rates. |
| ICT-026 | Story | Fill simulation | P1 | TODO | `backtest/fills.py` — model realistic order fills: market orders fill at bid/ask + slippage, limit orders may not fill (probability based on distance from market), partial fills for large orders relative to typical volume. |
| ICT-027 | Task | Market order fill model | P1 | TODO | Market buy fills at ask + slippage. Market sell fills at bid - slippage. Immediate execution assumed. |
| ICT-028 | Task | Limit order fill probability | P2 | TODO | Limit buy fills only if price trades through order level. Fill probability decreases as order sits further from market. Configurable fill rate cap (default 85%). |
| ICT-029 | Story | Performance metrics calculator | P0 | TODO | `backtest/metrics.py` — compute: CAGR, Sharpe ratio, Sortino ratio, max drawdown (depth + duration), win rate, avg win/loss ratio, profit factor, total trades, expectancy, Calmar ratio. |
| ICT-030 | Task | Equity curve tracking | P1 | TODO | Track mark-to-market equity at every bar. Compute running drawdown. Store equity curve for charting. |
| ICT-031 | Task | Backtest results storage | P1 | TODO | Store each backtest run in `backtest_runs` table (params, date range, strategy) and results in `backtest_results` (all metrics). Store individual trades in `backtest_trades`. |
| ICT-032 | Story | Margin & leverage tracking | P2 | TODO | Track margin utilization per bar. Flag if margin level drops below configurable threshold (e.g., 100%). Simulate stop-out if margin level < 50%. |

**Acceptance:** Can run a full backtest offline on historical data. Results include all costs (spread, commission, slippage, swap). All key metrics computed. Results stored in DB.

---

## PHASE 3: ROBUSTNESS TESTING (Trust)
### Epic: ICT-3 — Backtest Robustness & Overfitting Detection

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-040 | Story | Out-of-sample testing | P0 | TODO | `backtest/overfitting.py` — split data into train (70%) and test (30%). Run strategy on both. Compare Sharpe, drawdown, profit factor. Flag if >30% degradation. |
| ICT-041 | Task | Train/test split utility | P1 | TODO | Configurable split ratio. Ensure split respects time ordering (no look-ahead). Gap between train/test optional (avoid data leakage). |
| ICT-042 | Story | Walk-forward analysis | P0 | TODO | `backtest/walk_forward.py` — rolling window: optimize on N days, test on M days, slide forward. Collect out-of-sample results across all windows. Report: % profitable windows, aggregate Sharpe, consistency score. |
| ICT-043 | Task | Rolling window optimizer | P1 | TODO | For each window: run strategy with parameter grid on training set. Select best params. Apply to test set. Record results. |
| ICT-044 | Story | Monte Carlo parameter sensitivity | P1 | TODO | `backtest/monte_carlo.py` — sample N parameter sets (default 1000) from parameter space. Run backtest for each. Compute: % of params with Sharpe > threshold, parameter sensitivity map, robustness score. |
| ICT-045 | Task | Parameter space definition | P1 | TODO | Define ranges for: fast_ma (5-50), slow_ma (20-200), size_units, etc. Support uniform and normal sampling. |
| ICT-046 | Story | Bootstrap resampling | P2 | TODO | Resample trade returns with replacement (10,000 iterations). Compute confidence intervals for Sharpe, max drawdown, CAGR. Report 5th/50th/95th percentiles. |
| ICT-047 | Story | Market regime detection | P2 | TODO | `backtest/regime.py` — classify bars into regimes: trending-high-vol, trending-low-vol, ranging-high-vol, ranging-low-vol. Use ADX + ATR. Track strategy performance per regime. Flag if strategy only works in one regime. |
| ICT-048 | Task | ADX + ATR regime classifier | P2 | TODO | ADX > 25 = trending, ATR > median = high vol. Four quadrants. Label each bar with regime. |
| ICT-049 | Task | Regime performance breakdown | P2 | TODO | For each regime: compute separate Sharpe, win rate, profit factor. Flag if any regime has negative expectancy. |
| ICT-050 | Task | Overfitting score | P1 | TODO | Composite score: (1) train vs OOS degradation, (2) walk-forward consistency, (3) Monte Carlo robustness %, (4) regime diversity. Single 0-100 score where >70 = trustworthy. |

**Acceptance:** Backtest results include overfitting score. Walk-forward analysis runs across multiple windows. Monte Carlo produces robustness %. Strategy performance broken down by market regime.

---

## PHASE 4: RISK MANAGEMENT (Safety)
### Epic: ICT-4 — Position Sizing & Risk Controls

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-060 | Story | Dynamic position sizing | P1 | TODO | `backtest/position_sizer.py` — implement Kelly criterion (1/4 Kelly for safety), fixed fractional (risk X% per trade), and fixed lot sizing. Configurable per strategy. |
| ICT-061 | Task | Kelly criterion calculator | P2 | TODO | Based on running win rate and avg win/loss, compute optimal Kelly fraction. Use 1/4 Kelly as conservative default. |
| ICT-062 | Task | Fixed fractional position sizing | P1 | TODO | Risk X% of account per trade. Position size = (account * risk%) / (stop_loss_distance * pip_value). |
| ICT-063 | Story | Stop-loss / take-profit | P1 | TODO | Add SL/TP to strategy interface. Paper engine and backtest engine must respect SL/TP. Track in paper_positions table. |
| ICT-064 | Task | ATR-based stop-loss | P2 | TODO | Stop-loss at N x ATR from entry (default 1.5x ATR(14)). Dynamic, adapts to volatility. |
| ICT-065 | Story | Drawdown-based position reduction | P2 | TODO | If running drawdown exceeds threshold (e.g., 10%), reduce position size by 50%. If exceeds 15%, halt trading. Configurable thresholds. |
| ICT-066 | Task | Max exposure limits | P1 | TODO | Cap total exposure at configurable % of account (default 90% margin utilization). Prevent over-leveraging. |
| ICT-067 | Story | Trailing stop | P3 | TODO | Move stop-loss in direction of profit as trade moves favorably. Lock in gains. Configurable trail distance. |

**Acceptance:** Position sizes adapt to account balance and risk. SL/TP enforced in backtest and paper trading. Drawdown circuit breakers active.

---

## PHASE 5: NEWS INTEGRATION (Edge)
### Epic: ICT-5 — Economic Calendar & Trade Pausing

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-070 | Story | FMP economic calendar fetch | P1 | TODO | `news/fmp_calendar.py` — fetch upcoming economic events from Financial Modeling Prep API. Filter by currency (EUR, USD) and impact level (high). Store in `news_events` table. |
| ICT-071 | Task | Calendar API integration | P1 | TODO | GET request to FMP `/economic_calendar` endpoint. Parse response. Store: event_name, currency, impact, datetime_utc, actual, forecast, previous. |
| ICT-072 | Task | Periodic calendar refresh | P2 | TODO | Fetch calendar daily (or on daemon startup). Cache in DB. Refresh if stale. |
| ICT-073 | Story | Trade pause logic | P1 | TODO | `news/pause_logic.py` — before opening any position, check if high-impact news is within configured window (NEWS_PAUSE_BEFORE_MIN=15, NEWS_PAUSE_AFTER_MIN=30). If so, skip trade. Log reason. |
| ICT-074 | Task | Integrate pause logic into paper engine | P1 | TODO | Paper engine daemon calls `should_pause()` before every trade decision. If paused, log and skip. |
| ICT-075 | Task | Integrate pause logic into backtest engine | P2 | TODO | Backtest engine also respects news pauses (if news data available for backtest period). |
| ICT-076 | Task | News event logging & alerting | P3 | TODO | Send Telegram alert when entering/exiting news pause window. Log all pause events. |

**Acceptance:** High-impact news events are fetched and stored. Trading pauses automatically before/after events. Paper engine and backtest engine both respect pauses.

---

## PHASE 6: DASHBOARD & REPORTING (Visibility)
### Epic: ICT-6 — Enhanced Dashboard & Analytics

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-080 | Story | P&L chart on dashboard | P1 | TODO | Add equity curve chart (Plotly line chart) showing cumulative P&L over time. For both live paper trading and backtest results. |
| ICT-081 | Story | Signal history table | P2 | TODO | Add paginated table showing recent signals: timestamp, signal type, fast MA, slow MA, mid price. |
| ICT-082 | Story | Trade history table | P1 | TODO | Show all paper trades: timestamp, side, entry/exit price, P&L per trade, cumulative P&L. |
| ICT-083 | Story | Backtest results viewer | P1 | TODO | Select a backtest run and view: all metrics, equity curve, trade list, drawdown chart. |
| ICT-084 | Story | Monte Carlo visualization | P2 | TODO | Fan chart showing Monte Carlo equity paths (percentile bands). Histogram of terminal wealth. |
| ICT-085 | Story | Risk metrics panel | P2 | TODO | Live display: current drawdown, margin utilization, position size, next news event countdown. |
| ICT-086 | Task | News calendar view | P3 | TODO | Show upcoming economic events on dashboard with countdown timers. Highlight active pause windows. |
| ICT-087 | Task | Regime indicator | P3 | TODO | Show current market regime classification on dashboard. Historical regime timeline chart. |

**Acceptance:** Dashboard shows P&L, trades, signals, backtest results. Monte Carlo visualization works. Risk metrics visible at a glance.

---

## PHASE 7: LIVE PREPARATION (Deploy)
### Epic: ICT-7 — Production Readiness

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-090 | Story | Paper vs backtest comparison | P1 | TODO | After paper trading for N days, compare paper results against backtest predictions. Flag significant divergence (>20% Sharpe difference). |
| ICT-091 | Story | Kill switch | P0 | TODO | Telegram command or dashboard button to immediately flatten all positions and halt trading. |
| ICT-092 | Task | Alerting thresholds | P1 | TODO | Telegram alerts when: drawdown exceeds threshold, margin low, daemon stale, news pause activated, daily P&L limit hit. |
| ICT-093 | Task | Live readiness checklist | P1 | TODO | Automated checks before going live: OAuth tokens fresh, data flowing, backtest gates passed, news calendar loaded, Telegram working. |
| ICT-094 | Task | Token auto-refresh | P2 | TODO | cTrader OAuth tokens expire in ~30 days. Auto-refresh before expiry. Alert if refresh fails. |

**Acceptance:** Paper trading validates backtest results. Kill switch works. All alerts configured. Live readiness checklist passes.

---

## PHASE 8: DEEP PAPER TRADING (Fail-Closed Simulation)
### Epic: ICT-8 — Production-Grade Paper Trading Infrastructure

| ID | Type | Title | Priority | Status | Description |
|----|------|-------|----------|--------|-------------|
| ICT-100 | Story | Fail-closed paper simulator | P0 | DONE | `paper/simulator.py` — full simulation engine with realistic execution: spread from live bid/ask, broker commission, volatility-based slippage, swap rates at daily rollover, margin tracking, stop-out simulation. Every order goes through fail-closed gates (stale data, spread check, margin check). |
| ICT-101 | Task | Stale data rejection | P0 | DONE | If no tick received for >30s, all orders are rejected with STALE_DATA. Fail-closed: no trading on stale data. |
| ICT-102 | Task | Spread sanity gate | P0 | DONE | If current spread > 10 pips (configurable), orders rejected. Prevents trading during liquidity withdrawal. |
| ICT-103 | Task | Margin check & stop-out | P0 | DONE | Track margin used, margin free, margin level. Force-close at 50% margin level (configurable stop-out). |
| ICT-104 | Task | Swap rate simulation | P1 | DONE | Charges swap at 22:00 UTC daily rollover. Wednesday = triple swap (covers weekend). Accumulates per position. |
| ICT-105 | Task | SL/TP monitoring | P1 | DONE | Stop-loss and take-profit checked on every tick. LONG: SL at bid, TP at bid. SHORT: SL at ask, TP at ask. |
| ICT-106 | Task | Full audit logging | P0 | DONE | Every decision (order created, filled, rejected, SL/TP triggered, swap charged, stop-out) written to audit log and event_log table. |
| ICT-107 | Story | Deep paper runner | P1 | DONE | `paper/run_paper.py` — connects simulator to live quotes, applies strategy + news pauses + drawdown circuit breakers. Sends Telegram alerts on trade events. |
| ICT-108 | Story | Live readiness guards | P0 | DONE | `paper/live_guards.py` — automated gate checks before switching to live: min 7 days paper trading, positive Sharpe, min trades, data feed active, news calendar loaded, OAuth tokens valid. |
| ICT-109 | Task | Paper mode default | P1 | TODO | All strategies start in paper mode. Live mode requires explicit switch + passing all guards. |
| ICT-110 | Task | Paper vs live mode toggle | P2 | TODO | Dashboard toggle to switch modes (with guard checks and confirmation). |
| ICT-111 | Task | Gap handling in paper mode | P1 | DONE | Missing ticks, stale data → trading disabled automatically. Resumes when data flows again. |

**Acceptance:** Paper simulator runs with full cost modeling. All orders go through fail-closed gates. Audit trail for every decision. Live readiness guards block premature live deployment. Minimum 7 days paper + positive Sharpe required before live.

---

## Implementation Progress

| Phase | Status | Items Done | Items Total |
|-------|--------|------------|-------------|
| 0 — Stabilize | DONE | 7/7 | 7 |
| 1 — Data Pipeline | DONE | 8/8 | 8 |
| 2 — Backtest Engine | DONE | 13/13 | 13 |
| 3 — Robustness | DONE | 11/11 | 11 |
| 4 — Risk Management | DONE | 8/8 | 8 |
| 5 — News Integration | DONE | 7/7 | 7 |
| 6 — Dashboard | DONE | 8/8 | 8 |
| 7 — Live Prep | PARTIAL | 2/5 | 5 |
| 8 — Deep Paper Trading | DONE | 10/12 | 12 |
| **TOTAL** | | **74/79** | **79** |

## Remaining TODO

| ID | Title | Priority |
|----|-------|----------|
| ICT-091 | Kill switch (Telegram command + dashboard) | P0 |
| ICT-092 | Alerting thresholds (drawdown, margin, stale) | P1 |
| ICT-094 | Token auto-refresh | P2 |
| ICT-109 | Paper mode default enforcement | P1 |
| ICT-110 | Paper/live mode toggle in dashboard | P2 |
