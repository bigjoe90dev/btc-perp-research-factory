"""
Deep paper trading runner.

Connects the PaperSimulator to live market data from the quotes table.
Uses the strategy + news pause logic + position sizing.

Usage:
    python -m paper.run_paper
"""

import time
import os
from datetime import datetime, timezone, timedelta

from shared.config import load_settings
from shared.db import connect
from shared.logger import log_event
from backtest.costs import CostConfig
from backtest.position_sizer import SizingConfig, calculate_position_size, drawdown_action
from news.pause_logic import should_pause_trading
from paper.simulator import PaperSimulator, OrderSide, OrderType
from shared.glsb_strategy import GLSBConfig, GLSBStrategy
from shared.ma_cross_strategy import MACrossConfig, MACrossStrategy

try:
    from shared.telegram import send_telegram
except Exception:
    send_telegram = None


def fetch_latest_quote(db, env: str, account_id: str, symbol_id: int):
    row = db.execute(
        """SELECT ts_utc, bid, ask FROM quotes
           WHERE env=? AND account_id=? AND symbol_id=?
             AND bid IS NOT NULL AND ask IS NOT NULL
           ORDER BY id DESC LIMIT 1""",
        (env, account_id, symbol_id),
    ).fetchone()
    return row


def main():
    settings = load_settings(require_ctrader=False)

    env = (getattr(settings, "ctrader_env", None) or "demo").strip().lower()
    account_id = str(getattr(settings, "ctrader_account_id", "") or "")
    symbol_id = int(getattr(settings, "ctrader_symbol_id", 0) or 0)
    symbol_name = (getattr(settings, "ctrader_symbol_name", None) or "EURUSD").strip()

    if not account_id or not symbol_id:
        raise RuntimeError("Missing CTRADER_ACCOUNT_ID / CTRADER_SYMBOL_ID in .env")

    # Strategy selection
    strategy_key = (os.getenv("TRADING_STRATEGY") or "MA_CROSS").strip().upper()
    glsb_cfg = None
    ma_cfg = None
    use_news_pause = False
    if strategy_key in ("MA_CROSS", "EURUSD", "MA"):
        ma_cfg = MACrossConfig.from_env()
        strategy = MACrossStrategy(ma_cfg)
        strategy_label = "MA_CROSS"
        max_spread_usd = ma_cfg.max_spread
        risk_per_trade = ma_cfg.risk_per_trade
        max_trades_per_day = ma_cfg.max_trades_per_day
        max_daily_loss_pct = ma_cfg.max_daily_loss_pct
    else:
        glsb_cfg = GLSBConfig.from_env()
        strategy = GLSBStrategy(glsb_cfg)
        strategy_label = "GLSB_v1"
        max_spread_usd = glsb_cfg.max_spread_usd
        risk_per_trade = glsb_cfg.risk_per_trade
        max_trades_per_day = glsb_cfg.max_trades_per_day
        max_daily_loss_pct = glsb_cfg.max_daily_loss_pct
        use_news_pause = True

# Cost model (pip config from env)
    pip_pos = int(os.getenv("CTRADER_PIP_POSITION", "4"))
    pip_size = 10 ** (-pip_pos)
    pip_value = float(os.getenv("CTRADER_PIP_VALUE_PER_LOT", "10.0"))
    cost_cfg = CostConfig(pip_size=pip_size, pip_value_per_lot=pip_value)

    # Paper simulator
    sim = PaperSimulator(
        initial_balance=10_000.0,
        cost_config=cost_cfg,
        leverage=30.0,
        symbol=symbol_name,
        env=env,
        account_id=account_id,
        symbol_id=symbol_id,
        max_spread_pips=(max_spread_usd / cost_cfg.pip_size)
        if cost_cfg.pip_size else 10.0,
    )

    # Position sizing
    sizing_cfg = SizingConfig(
        method="fixed_fractional",
        risk_per_trade=risk_per_trade,
        pip_value_per_lot=pip_value,
    )

    db = connect()
    last_quote_ts = None

    log_event("INFO", "paper_runner",
              f"Deep paper trader started: env={env} symbol={symbol_name} "
              f"strategy={strategy_label}")

    if send_telegram:
        try:
            send_telegram(settings,
                          f"ict-bot PAPER: Deep simulator started ({symbol_name})")
        except Exception:
            pass

    print(f"Deep paper trader running: {symbol_name} {strategy_label}")
    print("Waiting for quotes...\n")

    while True:
        row = fetch_latest_quote(db, env, account_id, symbol_id)
        if not row:
            time.sleep(1)
            continue

        ts_utc_str, bid, ask = row["ts_utc"], float(row["bid"]), float(row["ask"])

        if ts_utc_str == last_quote_ts:
            time.sleep(1)
            continue
        last_quote_ts = ts_utc_str

        try:
            ts = datetime.fromisoformat(ts_utc_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            ts = datetime.now(timezone.utc)

        # Feed tick to simulator
        sim.on_tick(ts, bid, ask)

        # Clear active trade state if position closed externally (SL/TP)
        if sim.position is None and getattr(sim, "_glsb_active_entry", None) is not None:
            sim._glsb_active_entry = None
            sim._glsb_active_sl = None
            sim._glsb_active_side = None
            sim._glsb_active_session = None
            sim._glsb_active_range = None

        # Log any closed trade from simulator
        if sim.last_close:
            lc = sim.last_close
            msg = (f"PAPER CLOSE {symbol_name} {lc['side']} "
                   f"entry={lc['entry_price']:.2f} exit={lc['exit_price']:.2f} "
                   f"lots={lc['size_lots']:.2f} pnl=${lc['net_pnl']:.2f} "
                   f"reason={lc['reason']}")
            print(f"  {msg}")
            if send_telegram:
                try:
                    send_telegram(settings, f"ict-bot: {msg}")
                except Exception:
                    pass
            sim.last_close = None

        # Daily reset (session timezone)
        if glsb_cfg:
            session_day = (ts + timedelta(hours=glsb_cfg.tz_offset_hours)).date().isoformat()
        else:
            session_day = ts.date().isoformat()
        if getattr(sim, "_glsb_session_day", None) != session_day:
            sim._glsb_session_day = session_day
            sim._glsb_day_start_balance = sim.account.balance
            sim._glsb_daily_trades = 0
            sim._glsb_halt_day = False
            sim._glsb_active_entry = None
            sim._glsb_active_sl = None
            sim._glsb_active_side = None
            sim._glsb_active_session = None
            sim._glsb_active_range = None

        # Check news pause (GLSB only)
        paused = False
        if use_news_pause:
            pause_status = should_pause_trading()
            paused = pause_status.should_pause
            if paused:
                print(f"  [PAUSED] {pause_status.reason}")

        # Check drawdown circuit breaker
        if sim.account.equity < sim.account.balance:
            dd = (sim.account.balance - sim.account.equity) / sim.account.balance
        else:
            dd = 0.0
        action = drawdown_action(dd, sizing_cfg)
        if action == "HALT":
            print(f"  [HALTED] Drawdown circuit breaker: {dd:.1%}")
            sim._glsb_halt_day = True

        # Daily loss guard
        day_start = getattr(sim, "_glsb_day_start_balance", sim.account.balance)
        daily_loss = max(0.0, day_start - sim.account.balance)
        loss_pct = (daily_loss / day_start) if day_start else 0.0
        if loss_pct >= max_daily_loss_pct:
            sim._glsb_halt_day = True

        # Spread guard (USD)
        spread_ok = (ask - bid) <= max_spread_usd

        # Entry signals
        if glsb_cfg:
            signal = strategy.on_tick(
                ts, bid, ask,
                paused=paused,
                spread_ok=spread_ok,
                has_position=sim.position is not None,
            )
        else:
            signal = strategy.on_tick(
                ts, bid, ask,
                spread_ok=spread_ok,
            )

        if signal:
            try:
                mid = (bid + ask) / 2.0
                db.execute(
                    """INSERT INTO signals (ts_utc, env, account_id, symbol_id, signal, fast_ma, slow_ma, mid, note)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (ts.isoformat(), env, account_id, symbol_id,
                     signal.side, 0.0, 0.0, float(mid), f"{strategy_label.lower()}:{signal.reason}"),
                )
                db.commit()
            except Exception:
                pass

        if signal and sim.position is None and not getattr(sim, "_glsb_halt_day", False):
            if sim._glsb_daily_trades >= max_trades_per_day:
                time.sleep(1)
                continue

            entry_px = ask if signal.side == "BUY" else bid
            sl_dist = abs(entry_px - signal.sl_price)
            sl_pips = sl_dist / cost_cfg.pip_size if cost_cfg.pip_size else 0.0
            lots = calculate_position_size(
                sim.account.balance, sl_pips, dd, sizing_cfg)

            order = sim.submit_order(
                OrderSide.BUY if signal.side == "BUY" else OrderSide.SELL,
                lots,
                OrderType.MARKET,
                sl_price=signal.sl_price,
                tp_price=signal.tp_price,
            )
            if order.status.value == "FILLED":
                sim._glsb_daily_trades += 1
                sim._glsb_active_entry = order.fill_price
                sim._glsb_active_sl = signal.sl_price
                sim._glsb_active_side = signal.side
                sim._glsb_active_session = getattr(signal, "session", None)
                if glsb_cfg:
                    try:
                        state = strategy.session_state.get(signal.session)
                        if state and state.range_high is not None and state.range_low is not None:
                            sim._glsb_active_range = state.range_high - state.range_low
                        else:
                            sim._glsb_active_range = 0.0
                    except Exception:
                        sim._glsb_active_range = 0.0
                else:
                    sim._glsb_active_range = 0.0
                session_note = f" session={sim._glsb_active_session}" if sim._glsb_active_session else ""
                msg = (f"PAPER OPEN {signal.side} {symbol_name} @ {order.fill_price:.2f} "
                       f"(lots={lots:.2f} sl={signal.sl_price:.2f} tp={signal.tp_price:.2f} "
                       f"{session_note} {signal.reason})")
                print(f"  {msg}")
                if send_telegram:
                    try:
                        send_telegram(settings, f"ict-bot: {msg}")
                    except Exception:
                        pass
            else:
                print(f"  [REJECTED] {order.reject_reason}")

        # Time-based exit
        if sim.position:
            if glsb_cfg and strategy.should_time_exit(ts, getattr(sim, "_glsb_active_session", None)):
                pnl = sim.close_position("TIME_EXIT")
                print(f"  CLOSED (TIME_EXIT) -> pnl=${pnl:.2f}")
            elif ma_cfg and strategy.should_time_exit(ts):
                pnl = sim.close_position("TIME_EXIT")
                print(f"  CLOSED (TIME_EXIT) -> pnl=${pnl:.2f}")

        # Trailing / Breakeven management
        if sim.position and getattr(sim, "_glsb_active_entry", None) is not None:
            entry = sim._glsb_active_entry
            side = sim._glsb_active_side
            rng = sim._glsb_active_range or 0.0
            risk = abs(entry - sim._glsb_active_sl) if sim._glsb_active_sl else 0.0
            if glsb_cfg:
                if side == "BUY":
                    current = bid
                    if risk > 0 and current >= entry + (glsb_cfg.breakeven_rr * risk):
                        sim.position.sl_price = max(sim.position.sl_price, entry)
                    if rng > 0:
                        trail_sl = current - (rng * glsb_cfg.trail_pct_range)
                        sim.position.sl_price = max(sim.position.sl_price, trail_sl)
                elif side == "SELL":
                    current = ask
                    if risk > 0 and current <= entry - (glsb_cfg.breakeven_rr * risk):
                        sim.position.sl_price = min(sim.position.sl_price, entry)
                    if rng > 0:
                        trail_sl = current + (rng * glsb_cfg.trail_pct_range)
                        sim.position.sl_price = min(sim.position.sl_price, trail_sl)
            elif ma_cfg:
                if side == "BUY":
                    current = bid
                    if risk > 0 and current >= entry + risk:
                        sim.position.sl_price = max(sim.position.sl_price, entry)
                elif side == "SELL":
                    current = ask
                    if risk > 0 and current <= entry - risk:
                        sim.position.sl_price = min(sim.position.sl_price, entry)

        time.sleep(1)


if __name__ == "__main__":
    main()
