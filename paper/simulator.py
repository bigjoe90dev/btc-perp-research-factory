"""
Deep paper trading simulator — fail-closed execution engine.

Simulates realistic forex execution on cTrader including:
- Spread cost (live bid/ask)
- Broker commission
- Slippage (volatility-based)
- Swap rates (overnight financing)
- Margin tracking & stop-out simulation
- Gap handling (stale data / API downtime → halt trading)
- Full audit logging of every decision, order, fill, rejection

This is NOT the simple paper_engine_daemon.py. This is the production-grade
simulation that must pass before going live.
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from shared.db import connect
from shared.logger import log_event
from backtest.costs import CostConfig, apply_fill_price, slippage_pips


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class TradingMode(Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class Order:
    id: int
    ts_created: datetime
    side: OrderSide
    order_type: OrderType
    size_lots: float
    price: float = 0.0           # for limit/stop orders
    sl_price: float = 0.0        # stop-loss
    tp_price: float = 0.0        # take-profit
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_ts: Optional[datetime] = None
    filled_qty: float = 0.0
    reject_reason: str = ""
    slippage_pips: float = 0.0
    commission: float = 0.0


@dataclass
class PaperPosition:
    id: int
    side: str                     # "LONG" or "SHORT"
    size_lots: float
    entry_price: float
    entry_ts: datetime
    entry_weekday: int
    sl_price: float = 0.0
    tp_price: float = 0.0
    unrealized_pnl: float = 0.0
    swap_accumulated: float = 0.0
    last_swap_date: str = ""      # date of last swap charge


@dataclass
class AccountState:
    balance: float                # cash balance (after realized P&L)
    equity: float                 # balance + unrealized P&L
    margin_used: float = 0.0
    margin_free: float = 0.0
    margin_level: float = 999.0   # equity / margin_used * 100
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0
    total_swap: float = 0.0
    total_slippage: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


@dataclass
class AuditEntry:
    ts: datetime
    event_type: str               # ORDER_CREATED, ORDER_FILLED, ORDER_REJECTED, etc.
    details: dict


# ──────────────────────────────────────────────
# Paper Trading Simulator
# ──────────────────────────────────────────────

class PaperSimulator:
    """
    Production-grade paper trading simulator.

    Fail-closed design:
    - If data is stale → reject all orders
    - If margin insufficient → reject order
    - If spread is extreme → flag and optionally reject
    - Every action is audit-logged
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        cost_config: CostConfig = None,
        leverage: float = 30.0,
        symbol: str = "EURUSD",
        env: str = "demo",
        account_id: str = "paper_sim",
        symbol_id: int = 0,
        stale_data_threshold_sec: float = 30.0,
        max_spread_pips: float = 10.0,
    ):
        self.cfg = cost_config or CostConfig()
        self.leverage = leverage
        self.symbol = symbol
        self.env = env
        self.account_id = account_id
        self.symbol_id = symbol_id
        self.stale_threshold = stale_data_threshold_sec
        self.max_spread_pips = max_spread_pips

        # Account state
        self.account = AccountState(
            balance=initial_balance,
            equity=initial_balance,
            margin_free=initial_balance,
        )

        # Position tracking (one position at a time for now)
        self.position: Optional[PaperPosition] = None
        self.position_counter = 0

        # Order tracking
        self.orders: List[Order] = []
        self.order_counter = 0

        # Audit log
        self.audit_log: List[AuditEntry] = []

        # Market state
        self.last_bid: float = 0.0
        self.last_ask: float = 0.0
        self.last_tick_ts: Optional[datetime] = None
        self.recent_mids: deque = deque(maxlen=50)

        # Session stats
        self.mode = TradingMode.PAPER

        # Last closed trade snapshot (for clear logging)
        self.last_close: Optional[dict] = None

    # ──────────────────────────────────────────
    # Market data feed
    # ──────────────────────────────────────────

    def on_tick(self, ts: datetime, bid: float, ask: float):
        """
        Process incoming tick. Updates market state, checks SL/TP,
        applies swap if new day, updates margin.
        """
        self.last_bid = bid
        self.last_ask = ask
        self.last_tick_ts = ts
        self.recent_mids.append((bid + ask) / 2)

        # Check for overnight swap
        self._check_swap(ts)

        # Check SL/TP
        self._check_sl_tp(ts, bid, ask)

        # Update account state
        self._update_account(bid, ask)

        # Check margin stop-out
        self._check_stop_out(ts, bid, ask)

    def is_data_stale(self, now: datetime = None) -> bool:
        """Check if market data is stale (no ticks for too long)."""
        if self.last_tick_ts is None:
            return True
        if now is None:
            now = datetime.now(timezone.utc)
        age = (now - self.last_tick_ts).total_seconds()
        return age > self.stale_threshold

    def current_spread_pips(self) -> float:
        if self.last_bid <= 0 or self.last_ask <= 0:
            return 999.0
        return (self.last_ask - self.last_bid) / self.cfg.pip_size

    # ──────────────────────────────────────────
    # Order submission (fail-closed)
    # ──────────────────────────────────────────

    def submit_order(self, side: OrderSide, size_lots: float,
                     order_type: OrderType = OrderType.MARKET,
                     price: float = 0.0,
                     sl_price: float = 0.0,
                     tp_price: float = 0.0) -> Order:
        """
        Submit an order. Returns Order with status (FILLED or REJECTED).

        Fail-closed checks:
        1. Data freshness
        2. Spread sanity
        3. Margin adequacy
        4. Position conflicts
        """
        ts = datetime.now(timezone.utc)
        self.order_counter += 1

        order = Order(
            id=self.order_counter,
            ts_created=ts,
            side=side,
            order_type=order_type,
            size_lots=size_lots,
            price=price,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        # Gate 1: Data freshness
        if self.is_data_stale(ts):
            order.status = OrderStatus.REJECTED
            order.reject_reason = "STALE_DATA"
            self._audit("ORDER_REJECTED", {
                "order_id": order.id, "reason": "Data is stale",
                "last_tick_age": self._tick_age(ts),
            })
            self.orders.append(order)
            return order

        # Gate 2: Spread check
        spread = self.current_spread_pips()
        if spread > self.max_spread_pips:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"SPREAD_TOO_WIDE ({spread:.1f} pips)"
            self._audit("ORDER_REJECTED", {
                "order_id": order.id, "reason": f"Spread {spread:.1f} > {self.max_spread_pips}",
            })
            self.orders.append(order)
            return order

        # Gate 3: Margin check
        notional = size_lots * 100_000
        margin_required = notional / self.leverage
        if margin_required > self.account.margin_free:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "INSUFFICIENT_MARGIN"
            self._audit("ORDER_REJECTED", {
                "order_id": order.id,
                "margin_required": margin_required,
                "margin_free": self.account.margin_free,
            })
            self.orders.append(order)
            return order

        # Gate 4: Position conflict (close existing opposite before opening)
        if self.position is not None:
            if (side == OrderSide.BUY and self.position.side == "SHORT") or \
               (side == OrderSide.SELL and self.position.side == "LONG"):
                self._close_position(ts, "SIGNAL_FLIP")

        if self.position is not None and order_type == OrderType.MARKET:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "POSITION_ALREADY_OPEN"
            self._audit("ORDER_REJECTED", {
                "order_id": order.id, "reason": "Already have open position",
            })
            self.orders.append(order)
            return order

        # Execute market order
        if order_type == OrderType.MARKET:
            self._execute_market_order(order, ts)
        else:
            # Limit/stop orders queued (pending)
            self._audit("ORDER_CREATED", {
                "order_id": order.id, "type": order_type.value,
                "side": side.value, "price": price,
            })

        self.orders.append(order)
        return order

    def close_position(self, reason: str = "MANUAL") -> Optional[float]:
        """Close current position. Returns realized P&L or None."""
        if self.position is None:
            return None
        ts = datetime.now(timezone.utc)
        return self._close_position(ts, reason)

    # ──────────────────────────────────────────
    # Internal execution
    # ──────────────────────────────────────────

    def _execute_market_order(self, order: Order, ts: datetime):
        """Fill a market order at current bid/ask + slippage."""
        atr_pips = self._current_atr_pips()
        side_str = "LONG" if order.side == OrderSide.BUY else "SHORT"

        fill_price = apply_fill_price(side_str, self.last_bid, self.last_ask,
                                      atr_pips, self.cfg)
        slip = slippage_pips(atr_pips, self.cfg)

        # Commission (one side; other side charged on close)
        comm = self.cfg.commission_per_lot_per_side * order.size_lots

        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_ts = ts
        order.filled_qty = order.size_lots
        order.slippage_pips = slip
        order.commission = comm

        # Open position
        self.position_counter += 1
        self.position = PaperPosition(
            id=self.position_counter,
            side=side_str,
            size_lots=order.size_lots,
            entry_price=fill_price,
            entry_ts=ts,
            entry_weekday=ts.weekday(),
            sl_price=order.sl_price,
            tp_price=order.tp_price,
        )

        self.account.total_commission += comm
        self.account.total_slippage += slip * self.cfg.pip_value_per_lot * order.size_lots
        self.account.balance -= comm  # deduct entry commission
        self.account.total_trades += 1

        self._audit("ORDER_FILLED", {
            "order_id": order.id, "side": side_str,
            "fill_price": fill_price, "size_lots": order.size_lots,
            "slippage_pips": slip, "commission": comm,
            "bid": self.last_bid, "ask": self.last_ask,
        })

        # Persist to DB
        self._persist_trade("OPEN", side_str, order.size_lots, fill_price, ts,
                            f"slip={slip:.2f}pips comm=${comm:.2f}")

    def _close_position(self, ts: datetime, reason: str) -> float:
        """Close current position, compute P&L, update account."""
        pos = self.position
        if pos is None:
            return 0.0

        atr_pips = self._current_atr_pips()
        exit_side = "SELL" if pos.side == "LONG" else "BUY"
        exit_price = apply_fill_price(exit_side, self.last_bid, self.last_ask,
                                      atr_pips, self.cfg)

        # Gross P&L
        if pos.side == "LONG":
            price_diff = exit_price - pos.entry_price
        else:
            price_diff = pos.entry_price - exit_price

        pips = price_diff / self.cfg.pip_size
        gross_pnl = pips * self.cfg.pip_value_per_lot * pos.size_lots

        # Exit costs
        exit_comm = self.cfg.commission_per_lot_per_side * pos.size_lots
        exit_slip = slippage_pips(atr_pips, self.cfg) * self.cfg.pip_value_per_lot * pos.size_lots

        net_pnl = gross_pnl - exit_comm - exit_slip - pos.swap_accumulated

        # Update account
        self.account.balance += net_pnl
        self.account.total_commission += exit_comm
        self.account.total_slippage += exit_slip
        self.account.total_swap += pos.swap_accumulated

        if net_pnl > 0:
            self.account.winning_trades += 1
        elif net_pnl < 0:
            self.account.losing_trades += 1

        self._audit("POSITION_CLOSED", {
            "position_id": pos.id, "side": pos.side,
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "gross_pnl": gross_pnl, "net_pnl": net_pnl,
            "exit_commission": exit_comm, "swap_total": pos.swap_accumulated,
            "reason": reason,
        })

        self._persist_trade("CLOSE", pos.side, pos.size_lots, exit_price, ts,
                            f"reason={reason} pnl=${net_pnl:.2f}")

        self.last_close = {
            "ts": ts.isoformat(),
            "side": pos.side,
            "size_lots": pos.size_lots,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "net_pnl": net_pnl,
            "reason": reason,
        }

        self.position = None
        return net_pnl

    def _check_sl_tp(self, ts: datetime, bid: float, ask: float):
        """Check if stop-loss or take-profit has been hit."""
        pos = self.position
        if pos is None:
            return

        if pos.side == "LONG":
            # SL: bid drops to/below SL
            if pos.sl_price > 0 and bid <= pos.sl_price:
                self._audit("SL_TRIGGERED", {
                    "position_id": pos.id, "sl_price": pos.sl_price, "bid": bid,
                })
                self._close_position(ts, "STOP_LOSS")
                return
            # TP: bid rises to/above TP
            if pos.tp_price > 0 and bid >= pos.tp_price:
                self._audit("TP_TRIGGERED", {
                    "position_id": pos.id, "tp_price": pos.tp_price, "bid": bid,
                })
                self._close_position(ts, "TAKE_PROFIT")
                return

        elif pos.side == "SHORT":
            # SL: ask rises to/above SL
            if pos.sl_price > 0 and ask >= pos.sl_price:
                self._audit("SL_TRIGGERED", {
                    "position_id": pos.id, "sl_price": pos.sl_price, "ask": ask,
                })
                self._close_position(ts, "STOP_LOSS")
                return
            # TP: ask drops to/below TP
            if pos.tp_price > 0 and ask <= pos.tp_price:
                self._audit("TP_TRIGGERED", {
                    "position_id": pos.id, "tp_price": pos.tp_price, "ask": ask,
                })
                self._close_position(ts, "TAKE_PROFIT")
                return

    def _check_swap(self, ts: datetime):
        """Apply swap charges at daily rollover (22:00 UTC)."""
        pos = self.position
        if pos is None:
            return

        current_date = ts.strftime("%Y-%m-%d")
        if current_date == pos.last_swap_date:
            return  # already charged today

        # Check if we've crossed rollover (22:00 UTC)
        if ts.hour >= 22:
            rate = (self.cfg.swap_long_pips_per_day
                    if pos.side == "LONG"
                    else self.cfg.swap_short_pips_per_day)

            # Wednesday = triple swap
            multiplier = 3 if ts.weekday() == self.cfg.triple_swap_day else 1

            swap_cost = abs(rate) * self.cfg.pip_value_per_lot * pos.size_lots * multiplier
            pos.swap_accumulated += swap_cost
            pos.last_swap_date = current_date

            self._audit("SWAP_CHARGED", {
                "position_id": pos.id, "swap_cost": swap_cost,
                "multiplier": multiplier, "total_swap": pos.swap_accumulated,
            })

    def _check_stop_out(self, ts: datetime, bid: float, ask: float):
        """Force-close position if margin level drops below stop-out."""
        if self.position is None:
            return
        if self.account.margin_level < self.cfg.margin_stop_out_level * 100:
            self._audit("STOP_OUT", {
                "margin_level": self.account.margin_level,
                "threshold": self.cfg.margin_stop_out_level * 100,
            })
            self._close_position(ts, "MARGIN_STOP_OUT")

    def _update_account(self, bid: float, ask: float):
        """Recalculate account equity, margin, unrealized P&L."""
        pos = self.position
        if pos is None:
            self.account.equity = self.account.balance
            self.account.unrealized_pnl = 0.0
            self.account.margin_used = 0.0
            self.account.margin_free = self.account.balance
            self.account.margin_level = 999.0
            return

        # Unrealized P&L
        if pos.side == "LONG":
            price_diff = bid - pos.entry_price
        else:
            price_diff = pos.entry_price - ask

        pips = price_diff / self.cfg.pip_size
        pos.unrealized_pnl = pips * self.cfg.pip_value_per_lot * pos.size_lots

        self.account.unrealized_pnl = pos.unrealized_pnl
        self.account.equity = self.account.balance + pos.unrealized_pnl

        # Margin
        notional = pos.size_lots * 100_000
        self.account.margin_used = notional / self.leverage
        self.account.margin_free = self.account.equity - self.account.margin_used

        if self.account.margin_used > 0:
            self.account.margin_level = (self.account.equity / self.account.margin_used) * 100
        else:
            self.account.margin_level = 999.0

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _current_atr_pips(self) -> float:
        """Estimate ATR from recent mid prices."""
        if len(self.recent_mids) < 2:
            return 1.0
        vals = list(self.recent_mids)
        changes = [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        atr = sum(changes) / len(changes)
        return atr / self.cfg.pip_size

    def _tick_age(self, now: datetime) -> float:
        if self.last_tick_ts is None:
            return 9999.0
        return (now - self.last_tick_ts).total_seconds()

    def _audit(self, event_type: str, details: dict):
        """Record audit entry."""
        ts = datetime.now(timezone.utc)
        self.audit_log.append(AuditEntry(ts=ts, event_type=event_type, details=details))

        # Also write to event_log table
        try:
            log_event("INFO", "paper_sim", f"{event_type}: {json.dumps(details, default=str)}",
                      echo=False)
        except Exception:
            pass

    def _persist_trade(self, action: str, side: str, size_lots: float,
                       price: float, ts: datetime, note: str = ""):
        """Write trade to paper_trades table for dashboard/audit."""
        try:
            db = connect()
            db.execute(
                """INSERT INTO paper_trades
                   (ts_utc, env, account_id, symbol_id, action, side, size_units, mid, note)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (ts.isoformat(), self.env, self.account_id, int(self.symbol_id),
                 action, side, size_lots, price, note),
            )
            db.commit()
            db.close()
        except Exception:
            pass

    # ──────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────

    def account_summary(self) -> str:
        a = self.account
        lines = [
            "=== Paper Trading Account ===",
            f"  Mode:            {self.mode.value}",
            f"  Balance:         ${a.balance:,.2f}",
            f"  Equity:          ${a.equity:,.2f}",
            f"  Unrealized P&L:  ${a.unrealized_pnl:+,.2f}",
            f"  Margin Used:     ${a.margin_used:,.2f}",
            f"  Margin Free:     ${a.margin_free:,.2f}",
            f"  Margin Level:    {a.margin_level:.0f}%",
            "",
            f"  Total Trades:    {a.total_trades}",
            f"  Wins:            {a.winning_trades}",
            f"  Losses:          {a.losing_trades}",
            f"  Total Commission:${a.total_commission:,.2f}",
            f"  Total Swap:      ${a.total_swap:,.2f}",
            f"  Total Slippage:  ${a.total_slippage:,.2f}",
        ]
        return "\n".join(lines)

    def audit_summary(self, last_n: int = 20) -> str:
        entries = self.audit_log[-last_n:]
        lines = [f"=== Audit Log (last {last_n}) ==="]
        for e in entries:
            lines.append(f"  [{e.ts.strftime('%H:%M:%S')}] {e.event_type}: {e.details}")
        return "\n".join(lines)
