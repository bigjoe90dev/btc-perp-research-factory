import time
from collections import deque

from shared.db import connect
from shared.config import load_settings

try:
    from shared.telegram import send_telegram
except Exception:
    send_telegram = None


def now_utc_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def fetch_latest_quote(db, env: str, account_id: str, symbol_id: int):
    row = db.execute(
        """
        SELECT ts_utc, bid, ask
        FROM quotes
        WHERE env=? AND account_id=? AND symbol_id=? AND bid IS NOT NULL AND ask IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (env, account_id, symbol_id),
    ).fetchone()
    return row


def get_open_position(db, env: str, account_id: str, symbol_id: int):
    return db.execute(
        """
        SELECT id, side, size_units, entry_mid
        FROM paper_positions
        WHERE env=? AND account_id=? AND symbol_id=? AND status='OPEN'
        ORDER BY id DESC
        LIMIT 1
        """,
        (env, account_id, symbol_id),
    ).fetchone()


def open_position(db, env, account_id, symbol_id, side, size_units, mid, note=""):
    ts = now_utc_iso()
    db.execute(
        """
        INSERT INTO paper_positions (ts_utc, env, account_id, symbol_id, side, size_units, entry_mid, status)
        VALUES (?,?,?,?,?,?,?, 'OPEN')
        """,
        (ts, env, account_id, symbol_id, side, float(size_units), float(mid)),
    )
    db.execute(
        """
        INSERT INTO paper_trades (ts_utc, env, account_id, symbol_id, action, side, size_units, mid, note)
        VALUES (?,?,?,?, 'OPEN', ?,?,?,?)
        """,
        (ts, env, account_id, symbol_id, side, float(size_units), float(mid), note),
    )
    db.commit()
    return ts


def close_position(db, env, account_id, symbol_id, pos_id, side, size_units, mid, note=""):
    ts = now_utc_iso()
    db.execute(
        "UPDATE paper_positions SET status='CLOSED' WHERE id=?",
        (pos_id,),
    )
    db.execute(
        """
        INSERT INTO paper_trades (ts_utc, env, account_id, symbol_id, action, side, size_units, mid, note)
        VALUES (?,?,?,?, 'CLOSE', ?,?,?,?)
        """,
        (ts, env, account_id, symbol_id, side, float(size_units), float(mid), note),
    )
    db.commit()
    return ts


def moving_average(values, n: int):
    if len(values) < n:
        return None
    return sum(list(values)[-n:]) / n


def main():
    settings = load_settings(require_ctrader=False)

    env = (getattr(settings, "ctrader_env", None) or "demo").strip().lower()
    account_id = str(getattr(settings, "ctrader_account_id", "") or "")
    symbol_id = int(getattr(settings, "ctrader_symbol_id", 0) or 0)
    symbol_name = (getattr(settings, "ctrader_symbol_name", None) or str(symbol_id)).strip()

    if not account_id or not symbol_id:
        raise RuntimeError("Missing CTRADER_ACCOUNT_ID / CTRADER_SYMBOL_ID in .env")

    # simple defaults (can move to .env later)
    fast_n = int(getattr(settings, "paper_fast_ma", 10) or 10)   # ~10 seconds if 1Hz loop
    slow_n = int(getattr(settings, "paper_slow_ma", 30) or 30)
    size_units = float(getattr(settings, "paper_size_units", 10000) or 10000)

    db = connect()
    mids = deque(maxlen=max(fast_n, slow_n) + 5)

    last_quote_ts = None
    last_signal = None

    print(f"Paper engine ✅ env={env} account={account_id} symbol={symbol_name} ({symbol_id})")
    print(f"Strategy: MA cross fast={fast_n} slow={slow_n} | size={size_units} units")
    print("Running… (Ctrl+C to stop)\n")

    while True:
        row = fetch_latest_quote(db, env, account_id, symbol_id)
        if not row:
            time.sleep(1)
            continue

        ts_utc, bid, ask = row
        if ts_utc == last_quote_ts:
            time.sleep(1)
            continue

        last_quote_ts = ts_utc
        mid = (float(bid) + float(ask)) / 2.0
        mids.append(mid)

        fast = moving_average(mids, fast_n)
        slow = moving_average(mids, slow_n)

        if fast is None or slow is None:
            time.sleep(1)
            continue

        # signal
        if fast > slow:
            signal = "BUY"
        elif fast < slow:
            signal = "SELL"
        else:
            signal = "FLAT"

        # log signal only if it changes
        if signal != last_signal:
            last_signal = signal
            db.execute(
                """
                INSERT INTO signals (ts_utc, env, account_id, symbol_id, signal, fast_ma, slow_ma, mid, note)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (now_utc_iso(), env, account_id, symbol_id, signal, float(fast), float(slow), float(mid), "ma_cross"),
            )
            db.commit()
            print(f"[{ts_utc}] signal={signal} mid={mid:.5f} fast={fast:.5f} slow={slow:.5f}")

        # position logic (one position max)
        pos = get_open_position(db, env, account_id, symbol_id)

        if signal == "BUY":
            if not pos:
                open_position(db, env, account_id, symbol_id, "LONG", size_units, mid, note="ma_cross")
                msg = f"ict-bot (PAPER) ✅ OPEN LONG {symbol_name} @ {mid:.5f}"
                print(msg)
                if send_telegram:
                    try:
                        send_telegram(settings, msg)
                    except Exception:
                        pass
            elif pos[1] == "SHORT":
                close_position(db, env, account_id, symbol_id, pos[0], pos[1], pos[2], mid, note="flip_to_long")
                open_position(db, env, account_id, symbol_id, "LONG", size_units, mid, note="ma_cross_flip")
        elif signal == "SELL":
            if not pos:
                open_position(db, env, account_id, symbol_id, "SHORT", size_units, mid, note="ma_cross")
                msg = f"ict-bot (PAPER) ✅ OPEN SHORT {symbol_name} @ {mid:.5f}"
                print(msg)
                if send_telegram:
                    try:
                        send_telegram(settings, msg)
                    except Exception:
                        pass
            elif pos[1] == "LONG":
                close_position(db, env, account_id, symbol_id, pos[0], pos[1], pos[2], mid, note="flip_to_short")
                open_position(db, env, account_id, symbol_id, "SHORT", size_units, mid, note="ma_cross_flip")

        time.sleep(1)


if __name__ == "__main__":
    main()
