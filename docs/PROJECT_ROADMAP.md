# ICT-Bot Project Roadmap
## Algorithmic Forex Trading Bot — cTrader + ICT Methodology

**Version:** 1.0
**Last Updated:** 2026-02-06
**Status:** Active Development

---

## Vision

Build a production-grade algorithmic forex trading bot that:
1. Streams live market data from cTrader (demo → live)
2. Generates trading signals based on ICT methodology / MA crossover
3. Backtests strategies with institutional-grade robustness (realistic costs, Monte Carlo, overfitting detection)
4. Paper trades to validate before deploying real capital
5. Monitors everything via dashboard + Telegram alerts
6. Pauses around high-impact news events

**Acceptance for live deployment:** Backtest must pass all robustness gates (see `BACKTEST_SPEC.md`)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ict-bot Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DATA LAYER                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ cTrader API   │  │ FMP News API │  │ CSV/History  │      │
│  │ (Live Quotes) │  │ (Calendar)   │  │ (Backtest)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         ▼                  ▼                  ▼               │
│  ┌─────────────────────────────────────────────────┐        │
│  │              SQLite (ict_bot.sqlite3)             │        │
│  │  quotes │ signals │ paper_trades │ paper_positions│        │
│  │  heartbeat │ event_log │ backtest_results         │        │
│  │  news_events │ backtest_runs │ regime_labels      │        │
│  └──────────────────────┬──────────────────────────┘        │
│                          │                                    │
│  ENGINE LAYER            │                                    │
│  ┌───────────┐  ┌───────┴──────┐  ┌──────────────┐         │
│  │ Market     │  │ Paper Engine │  │ Backtest     │         │
│  │ Data       │  │ (Live Sim)   │  │ Engine       │         │
│  │ Daemon     │  │              │  │ (Offline)    │         │
│  └───────────┘  └──────────────┘  └──────────────┘         │
│                                                              │
│  ROBUSTNESS LAYER                                            │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Cost Model │  │ Monte Carlo  │  │ Walk-Forward │         │
│  │ (Spread,   │  │ (Param       │  │ Analysis     │         │
│  │  Slippage, │  │  Sensitivity)│  │              │         │
│  │  Swap)     │  │              │  │              │         │
│  └───────────┘  └──────────────┘  └──────────────┘         │
│                                                              │
│  RISK LAYER                                                  │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Position   │  │ Drawdown     │  │ News Pause   │         │
│  │ Sizing     │  │ Controls     │  │ Logic        │         │
│  │ (Kelly)    │  │              │  │              │         │
│  └───────────┘  └──────────────┘  └──────────────┘         │
│                                                              │
│  PRESENTATION LAYER                                          │
│  ┌───────────────────────────┐  ┌──────────────────┐        │
│  │ Dash Dashboard (port 8050)│  │ Telegram Alerts  │        │
│  │ P&L, Signals, Backtest   │  │                   │        │
│  │ Monte Carlo, Risk Metrics │  │                   │        │
│  └───────────────────────────┘  └──────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase Summary

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| **0** | Stabilize | Not Started | Fix bugs, consolidate DB schema, validate daemons |
| **1** | Data Pipeline | Not Started | Data validation, historical data, integrity checks |
| **2** | Backtest Engine | Not Started | Core backtester with realistic forex costs |
| **3** | Robustness Testing | Not Started | Monte Carlo, walk-forward, overfitting detection |
| **4** | Risk Management | Not Started | Position sizing, stop-loss, drawdown controls |
| **5** | News Integration | Not Started | FMP economic calendar, trade pause logic |
| **6** | Dashboard & Reporting | Not Started | P&L charts, backtest viewer, risk metrics |
| **7** | Live Preparation | Not Started | Paper validation, kill switch, live readiness |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Database | SQLite3 (WAL mode) |
| Async/Networking | Twisted (cTrader Protobuf) |
| Trading API | cTrader Open API (Protobuf/TCP) |
| News API | Financial Modeling Prep (FMP) |
| Notifications | Telegram Bot API |
| Dashboard | Plotly Dash |
| Data Analysis | pandas, numpy, scipy |
| Auth | OAuth2 (cTrader / Spotware) |

---

## Key Metrics & Definitions (Forex Context)

| Metric | Definition |
|--------|-----------|
| **Spread Cost** | ask - bid at time of entry/exit (primary forex transaction cost) |
| **Commission** | Per-lot commission charged by cTrader broker (typically $3-7/lot round-trip) |
| **Slippage** | Difference between expected and actual fill price (0.1-0.5 pips for majors) |
| **Swap Rate** | Overnight financing charge for holding positions past rollover (broker-specific) |
| **Pip** | Smallest price increment = 0.0001 for EURUSD (4th decimal) |
| **Lot** | Standard lot = 100,000 units of base currency |
| **Margin** | Collateral required to open position (leverage-dependent) |
| **Stop-Out** | Broker force-closes position when margin level drops below threshold (typically 50%) |

---

## Risk Profile

- **Environment:** Demo (cTrader) — NO real money until all robustness gates pass
- **Symbol:** EURUSD (will expand to majors later)
- **Strategy:** MA Crossover (baseline) → ICT methodology (target)
- **Position Size:** 10,000 units (0.1 lot) — conservative for demo
- **Max Concurrent Positions:** 1

---

## File Structure (Target)

```
ict-bot/
├── shared/              # Shared utilities
│   ├── config.py        # Settings / env loading
│   ├── db.py            # Database connection
│   ├── telegram.py      # Telegram integration
│   └── logger.py        # Event logging (NEW)
├── daemon/              # Background services
│   ├── market_data_daemon.py    # Live quote streamer
│   ├── paper_engine_daemon.py   # Paper trading engine
│   └── heartbeat_writer.py      # Health monitor
├── backtest/            # Backtest engine (NEW)
│   ├── __init__.py
│   ├── engine.py        # Core backtest runner
│   ├── costs.py         # Commission, slippage, swap models
│   ├── fills.py         # Fill simulation
│   ├── metrics.py       # Performance metrics (Sharpe, etc.)
│   ├── walk_forward.py  # Walk-forward analysis
│   ├── monte_carlo.py   # Monte Carlo simulation
│   ├── overfitting.py   # Overfitting detection
│   ├── regime.py        # Market regime detection
│   ├── data_validator.py # Data integrity checks
│   └── position_sizer.py # Position sizing (Kelly, etc.)
├── news/                # News integration (NEW)
│   ├── __init__.py
│   ├── fmp_calendar.py  # FMP economic calendar fetch
│   └── pause_logic.py   # Trade pause around events
├── dashboard/           # Web dashboard
│   └── app.py           # Dash UI
├── scripts/             # Setup & utility scripts
├── data/                # Runtime data (SQLite, tokens)
├── docs/                # Documentation
│   ├── PROJECT_ROADMAP.md
│   ├── BACKLOG.md
│   └── BACKTEST_SPEC.md
├── tests/               # Test suite (NEW)
├── .env
├── .env.example
├── requirements.txt
└── README.md
```
