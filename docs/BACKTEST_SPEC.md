# Backtest Robustness Specification — ICT-Bot (Forex)
## Adapted from FR20 for cTrader / Forex

**Version:** 1.0
**Last Updated:** 2026-02-06
**Goal:** Produce backtests so robust that results are trustworthy for real money deployment on cTrader

---

## What We HAVE (Existing)

- [x] Live quote streaming from cTrader (EURUSD, 1-tick resolution)
- [x] Paper trading engine with MA crossover strategy
- [x] SQLite storage for quotes, signals, trades, positions
- [x] Telegram notifications on trade events
- [x] Basic dashboard with daemon health monitoring

## What We NEED (Critical Gaps)

---

### 1. Realistic Costs (CRITICAL)

**Why critical:** Forex trading has 3 cost layers. Ignoring spread + swap alone can turn a profitable backtest into a losing live bot.

#### 1a. Spread Cost
The bid-ask spread is the primary forex transaction cost. Unlike crypto with explicit maker/taker fees, forex costs are embedded in the spread.

```python
# Spread cost per trade
spread_pips = (ask - bid) / pip_size  # e.g., (1.08505 - 1.08490) / 0.0001 = 1.5 pips
spread_cost_per_lot = spread_pips * pip_value  # e.g., 1.5 * $10 = $15 per standard lot
```

**EURUSD typical spreads:**
- Raw spread: 0.1 - 0.3 pips (ECN broker)
- Standard spread: 1.0 - 1.5 pips (market maker)
- During news: 3 - 10+ pips (liquidity withdrawal)

#### 1b. Broker Commission
cTrader brokers charge per-lot commission on top of raw spread:

```python
# Per-trade commission
commission_per_lot_per_side = 3.50  # typical: $3.50 per lot per side
commission_round_trip = commission_per_lot_per_side * 2 * lots
```

#### 1c. Swap Rates (Overnight Financing)
Holding positions past daily rollover (17:00 EST / 22:00 UTC) incurs swap charges. Wednesday = triple swap (covers weekend).

```python
# Swap cost per day
swap_rate_long = -0.5   # pips per day (negative = cost)
swap_rate_short = -0.3  # pips per day
triple_swap_day = "Wednesday"

def calculate_swap_cost(side, lots, days_held, swap_rate, pip_value=10.0):
    # Wednesday positions get triple swap
    wednesday_count = count_wednesdays(days_held)
    effective_days = days_held + (wednesday_count * 2)  # triple = +2 extra
    return abs(swap_rate) * pip_value * lots * effective_days
```

**Why this matters:** A scalping strategy holding 30 minutes pays zero swap. A swing strategy holding 5 days on EURUSD pays ~$25-50 per lot in swap — this can erase thin edges.

#### 1d. Total Cost Model

```python
def total_trade_cost(entry_spread_pips, exit_spread_pips, slippage_pips,
                     commission_per_lot, lots, swap_cost):
    spread_cost = (entry_spread_pips + exit_spread_pips) / 2 * pip_value * lots
    slippage_cost = slippage_pips * pip_value * lots
    commission = commission_per_lot * 2 * lots  # round-trip
    return spread_cost + slippage_cost + commission + swap_cost
```

---

### 2. Slippage Simulation (CRITICAL)

**Why critical:** Backtest assumes fills at exact price. Reality: price moves between order submission and execution.

```python
def estimate_slippage(volatility_pips, order_type="market"):
    """
    Forex slippage model for major pairs.

    Components:
    - Fixed base: ~0.1 pips (execution latency)
    - Variable: proportional to recent volatility
    - News multiplier: 3-5x during high-impact events
    """
    base_slippage = 0.1  # pips, minimum execution cost
    vol_component = volatility_pips * 0.05  # 5% of recent ATR

    return base_slippage + vol_component

# During news events (already in our pause window, but for modeling)
def news_slippage_multiplier(minutes_from_event):
    if minutes_from_event < 2:
        return 5.0  # extreme slippage right at release
    elif minutes_from_event < 5:
        return 3.0
    elif minutes_from_event < 15:
        return 1.5
    return 1.0
```

---

### 3. Fill Simulation (CRITICAL)

**Why critical:** Backtest assumes all orders fill instantly at exact price. Real trading != this.

```python
def simulate_market_order_fill(side, bid, ask, slippage_pips, pip_size):
    """Market orders: fill at bid/ask + slippage."""
    if side == "BUY":
        fill_price = ask + (slippage_pips * pip_size)
    else:  # SELL
        fill_price = bid - (slippage_pips * pip_size)
    return fill_price

def simulate_limit_order_fill(order_price, bar_high, bar_low, side,
                               fill_rate_cap=0.85):
    """
    Limit orders: only fill if price trades through level.
    Not all limit orders fill even when price touches.
    """
    if side == "BUY" and bar_low <= order_price:
        # Price touched our buy level
        distance_through = (order_price - bar_low) / (bar_high - bar_low)
        fill_prob = min(fill_rate_cap, 0.5 + distance_through)
        return random() < fill_prob
    elif side == "SELL" and bar_high >= order_price:
        distance_through = (bar_high - order_price) / (bar_high - bar_low)
        fill_prob = min(fill_rate_cap, 0.5 + distance_through)
        return random() < fill_prob
    return False  # Price never reached our level
```

---

### 4. Monte Carlo Robustness Testing (CRITICAL)

```python
# Walk-forward analysis
def walk_forward(data, strategy_class, param_space,
                 train_days=90, test_days=30, step_days=30):
    results = []
    for train, test in rolling_windows(data, train_days, test_days, step_days):
        best_params = optimize_on(train, strategy_class, param_space)
        oos_result = backtest(test, strategy_class, best_params)
        results.append(oos_result)

    profitable_windows = sum(1 for r in results if r.profit_factor > 1.0)
    consistency = profitable_windows / len(results)
    return results, consistency

# Parameter sensitivity (Monte Carlo)
def monte_carlo_params(data, strategy_class, param_space, n_samples=1000):
    results = []
    for params in sample_params(param_space, n_samples):
        result = backtest(data, strategy_class, params)
        results.append((params, result))

    robust_count = sum(1 for _, r in results if r.sharpe > 1.0)
    robustness_score = robust_count / n_samples
    return results, robustness_score

# Bootstrap resampling
def bootstrap_confidence(trade_returns, n_iterations=10000):
    sharpes = []
    for _ in range(n_iterations):
        sample = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
        sharpes.append(calculate_sharpe(sample))

    return {
        "p5": np.percentile(sharpes, 5),
        "p50": np.percentile(sharpes, 50),
        "p95": np.percentile(sharpes, 95),
    }
```

---

### 5. Overfitting Detection (HIGH PRIORITY)

```python
def detect_overfitting(data, strategy_class, params, split_ratio=0.7):
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    oos_data = data[split_idx:]

    train_result = backtest(train_data, strategy_class, params)
    oos_result = backtest(oos_data, strategy_class, params)

    # Degradation metrics
    sharpe_degradation = (train_result.sharpe - oos_result.sharpe) / max(train_result.sharpe, 0.01)
    pf_degradation = (train_result.profit_factor - oos_result.profit_factor) / max(train_result.profit_factor, 0.01)
    dd_increase = (oos_result.max_drawdown - train_result.max_drawdown) / max(train_result.max_drawdown, 0.01)

    overfit_flags = []
    if sharpe_degradation > 0.30:
        overfit_flags.append(f"Sharpe degraded {sharpe_degradation:.0%}")
    if pf_degradation > 0.30:
        overfit_flags.append(f"Profit factor degraded {pf_degradation:.0%}")
    if dd_increase > 0.50:
        overfit_flags.append(f"Max drawdown increased {dd_increase:.0%}")

    return {
        "train_sharpe": train_result.sharpe,
        "oos_sharpe": oos_result.sharpe,
        "sharpe_degradation": sharpe_degradation,
        "overfit_flags": overfit_flags,
        "is_overfit": len(overfit_flags) > 0,
    }
```

---

### 6. Position Sizing & Risk Management (HIGH PRIORITY)

```python
def calculate_position_size(account_balance, risk_per_trade, stop_loss_pips,
                             pip_value=10.0, method="fixed_fractional"):
    if method == "fixed_fractional":
        # Risk X% of account per trade
        risk_amount = account_balance * risk_per_trade  # e.g., $10,000 * 0.01 = $100
        lots = risk_amount / (stop_loss_pips * pip_value)  # $100 / (20 pips * $10) = 0.5 lots
        return lots

    elif method == "kelly":
        win_rate = strategy.win_rate
        avg_win = strategy.avg_win_pips
        avg_loss = strategy.avg_loss_pips
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        # Conservative: 1/4 Kelly
        safe_fraction = max(0, kelly_fraction / 4)
        lots = (account_balance * safe_fraction) / (stop_loss_pips * pip_value)
        return lots

def drawdown_circuit_breaker(current_drawdown, max_allowed=0.15):
    """Reduce or halt trading based on drawdown."""
    if current_drawdown > max_allowed:
        return "HALT"  # Stop all trading
    elif current_drawdown > max_allowed * 0.67:
        return "REDUCE"  # Half position size
    return "NORMAL"
```

---

### 7. Market Regime Detection (MEDIUM PRIORITY)

```python
def detect_regime(prices, period=14):
    """Classify market into 4 regimes using ADX + ATR."""
    atr = calculate_atr(prices, period)
    adx = calculate_adx(prices, period)
    median_atr = np.median(atr[-100:])  # rolling median

    current_atr = atr[-1]
    current_adx = adx[-1]

    trending = current_adx > 25
    high_vol = current_atr > median_atr

    if trending and high_vol:
        return "trending_high_vol"    # Strong moves, wide ranges
    elif trending:
        return "trending_low_vol"     # Clean trends, tight ranges
    elif high_vol:
        return "ranging_high_vol"     # Choppy, whipsaw danger
    else:
        return "ranging_low_vol"      # Quiet, low opportunity
```

---

### 8. Data Integrity Validation (MEDIUM PRIORITY)

```python
def validate_quote_data(df, expected_interval_sec=1):
    """Validate quote data before running backtest."""
    issues = []

    # 1. Coverage check
    total_expected = (df['ts_utc'].max() - df['ts_utc'].min()).total_seconds() / expected_interval_sec
    coverage = len(df) / max(total_expected, 1)
    if coverage < 0.95:
        issues.append(f"CRITICAL: Only {coverage:.1%} bar coverage (expected >95%)")

    # 2. Missing bar gaps
    time_diffs = df['ts_utc'].diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(seconds=expected_interval_sec * 10)]
    if len(large_gaps) > 0:
        issues.append(f"WARNING: {len(large_gaps)} gaps > 10x expected interval")

    # 3. Outlier detection (extreme moves)
    df['mid'] = (df['bid'] + df['ask']) / 2
    returns = df['mid'].pct_change()
    extreme = returns[returns.abs() > 0.005]  # >0.5% move in one tick
    if len(extreme) > 0:
        issues.append(f"WARNING: {len(extreme)} extreme price moves (>0.5% per tick)")

    # 4. Zero spread / inverted spread
    df['spread'] = df['ask'] - df['bid']
    bad_spread = df[df['spread'] <= 0]
    if len(bad_spread) > 0:
        issues.append(f"CRITICAL: {len(bad_spread)} bars with zero/negative spread")

    # 5. Duplicates
    dupes = df.duplicated(subset=['ts_utc'], keep='first')
    if dupes.sum() > 0:
        issues.append(f"WARNING: {dupes.sum()} duplicate timestamps")

    return issues
```

---

## Acceptance Criteria (Robustness Gates)

### Backtest PASSES if ALL of the following are true:

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| G1 | Sharpe ratio (after all costs) | > 1.5 |
| G2 | Max drawdown | < 20% |
| G3 | Out-of-sample Sharpe within X% of in-sample | Within 30% |
| G4 | Walk-forward: % of windows profitable | > 80% |
| G5 | Profit factor (after all costs) | > 1.5 |
| G6 | No margin stop-outs in backtest period | 0 stop-outs |
| G7 | Data integrity: bar coverage | > 95% |
| G8 | Monte Carlo: % of param sets with Sharpe > 1.0 | > 30% |
| G9 | Strategy works in 2+ market regimes | Not single-regime |
| G10 | Bootstrap 5th percentile Sharpe | > 0.5 |

### Backtest FAILS if ANY of the following are true:

- Any gate criterion not met
- Strategy only profitable in one market regime
- >50% performance degradation out-of-sample
- >90% limit order fill rate assumed (unrealistic)
- Spread costs not modeled (results meaningless)
- <30 trades in test period (insufficient sample)

---

## What We DON'T Need (Avoid Over-Engineering)

| Skip | Reason |
|------|--------|
| Tick-by-tick order book simulation | Not HFT, 1s resolution sufficient |
| Multi-asset portfolio optimization | Single symbol (EURUSD) for now |
| ML-based slippage models | Fixed % slippage fine for majors |
| Intraday margin calls | Daily margin check sufficient |
| Cross-broker spread comparison | Single broker (cTrader demo) |
| Latency simulation | Not latency-sensitive at 1s resolution |

---

## Cost Defaults (EURUSD on cTrader)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `spread_typical_pips` | 1.2 | Average during London/NY session |
| `spread_news_pips` | 5.0 | During high-impact events |
| `commission_per_lot_per_side` | 3.50 | USD, typical ECN |
| `slippage_base_pips` | 0.1 | Fixed minimum |
| `slippage_vol_factor` | 0.05 | Multiplied by ATR |
| `swap_long_pips_per_day` | -0.5 | Negative = cost |
| `swap_short_pips_per_day` | -0.3 | Negative = cost |
| `triple_swap_day` | Wednesday | Industry standard |
| `pip_size` | 0.0001 | 4th decimal for EURUSD |
| `pip_value_per_lot` | 10.0 | USD per pip per standard lot |
| `limit_fill_rate_cap` | 0.85 | Max fill probability |
| `margin_stop_out_level` | 0.50 | 50% margin level |

---

## Estimated Implementation Effort

| Component | Effort |
|-----------|--------|
| Phase 0: Stabilize | ~0.5 day |
| Phase 1: Data Pipeline | ~1 day |
| Phase 2: Backtest Engine | ~3 days |
| Phase 3: Robustness Testing | ~2 days |
| Phase 4: Risk Management | ~1.5 days |
| Phase 5: News Integration | ~1 day |
| Phase 6: Dashboard | ~2 days |
| Phase 7: Live Prep | ~1 day |
| **Total** | **~12 days** |

---

## Key Insight

> The difference between "backtest works" and "bot makes money":
> 1. **Realistic costs** (spread + commission + slippage + swap)
> 2. **Realistic fills** (limit orders don't always fill)
> 3. **Overfitting detection** (out-of-sample + walk-forward)
>
> Without these 3, backtest is fiction.
