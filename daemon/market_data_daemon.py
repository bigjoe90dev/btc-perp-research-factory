import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf

from shared.config import load_settings
from shared.db import connect

from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOASubscribeSpotsReq,
    ProtoOASpotEvent,
)
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_access_token(tokens_path: Path) -> str:
    d = json.loads(tokens_path.read_text())
    token = d.get("access_token") or d.get("accessToken") or ""
    return (token or "").strip()


def _get_env_settings(settings):
    # Prefer Settings attrs if they exist, otherwise fall back to raw env vars.
    env = (getattr(settings, "ctrader_env", None) or os.getenv("CTRADER_ENV") or "demo").strip().lower()
    account_id = str(getattr(settings, "ctrader_account_id", None) or os.getenv("CTRADER_ACCOUNT_ID") or "").strip()
    symbol_id_raw = getattr(settings, "ctrader_symbol_id", None)
    if symbol_id_raw in (None, "", 0):
        symbol_id_raw = os.getenv("CTRADER_SYMBOL_ID") or ""
    symbol_id_raw = str(symbol_id_raw).strip()
    symbol_name = str(getattr(settings, "ctrader_symbol_name", None) or os.getenv("CTRADER_SYMBOL_NAME") or "").strip()

    digits_raw = getattr(settings, "ctrader_digits", None)
    if digits_raw in (None, ""):
        digits_raw = os.getenv("CTRADER_DIGITS") or "5"
    digits_raw = str(digits_raw).strip()

    if env not in ("demo", "live"):
        raise RuntimeError("CTRADER_ENV must be 'demo' or 'live'")

    if not account_id:
        raise RuntimeError("Missing CTRADER_ACCOUNT_ID in .env")

    if not symbol_id_raw:
        raise RuntimeError("Missing CTRADER_SYMBOL_ID in .env")

    try:
        symbol_id = int(symbol_id_raw)
    except ValueError:
        raise RuntimeError("CTRADER_SYMBOL_ID must be an integer (e.g. EURUSD is usually 1)")

    try:
        digits = int(digits_raw)
    except ValueError:
        digits = 5

    return env, account_id, symbol_id, symbol_name, digits


def _ensure_quotes_table():
    db = connect()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS quotes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts_utc TEXT NOT NULL,
          env TEXT NOT NULL,
          account_id TEXT NOT NULL,
          symbol_id INTEGER NOT NULL,
          bid REAL,
          ask REAL
        )
        """
    )
    db.commit()
    db.close()


def main():
    settings = load_settings(require_ctrader=False)

    env, account_id_str, symbol_id, symbol_name, digits = _get_env_settings(settings)

    client_id = (getattr(settings, "ctrader_client_id", "") or "").strip()
    client_secret = (getattr(settings, "ctrader_client_secret", "") or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env")

    access_token = _load_access_token(settings.tokens_path)
    if not access_token:
        raise RuntimeError(f"{settings.tokens_path} missing access token. Re-run scripts/ctrader_oauth_full.py")

    host = EndPoints.PROTOBUF_DEMO_HOST if env == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT

    _ensure_quotes_table()

    # Scaling + formatting
    scale = 10 ** max(digits, 0)
    scale_override = os.getenv("CTRADER_PRICE_DIVISOR", "").strip()
    if scale_override:
        try:
            scale = float(scale_override)
        except ValueError:
            pass
    fmt = f"{{:.{digits}f}}"

    # Zero-tick repair state
    last_bid = None
    last_ask = None

    # Write throttling: if you ever want “max 1 write per second”, set >0
    min_write_interval_sec = float(os.getenv("CTRADER_MIN_WRITE_INTERVAL_SEC", "0") or "0")
    last_write_ts = 0.0

    c = Client(host, port, TcpProtocol)
    print_raw = (os.getenv("CTRADER_PRINT_RAW", "1").strip().lower() in ("1", "true", "yes", "y", "on"))

    def stop():
        try:
            c.stopService()
        except Exception:
            pass
        try:
            reactor.stop()
        except Exception:
            pass

    def connected(_client):
        print(f"Connected ✅ env={env} accountId={account_id_str} symbolId={symbol_id}")
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        c.send(req)

    def disconnected(_client, reason):
        # This is normal when you Ctrl+C or the socket drops.
        print(f"Disconnected: {reason}")
        stop()

    def on_message(_client, msg):
        nonlocal last_bid, last_ask, last_write_ts

        try:
            payload = Protobuf.extract(msg)
        except Exception:
            return

        if isinstance(payload, ProtoErrorRes):
            print(f"❌ OpenAPI error: {getattr(payload, 'errorCode', None)} | {getattr(payload, 'description', None)}")
            stop()
            return

        if isinstance(payload, ProtoOAApplicationAuthRes):
            print("App auth OK ✅")
            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = int(account_id_str)
            req.accessToken = access_token
            c.send(req)
            return

        if isinstance(payload, ProtoOAAccountAuthRes):
            print("Account auth OK ✅")
            sub = ProtoOASubscribeSpotsReq()
            sub.ctidTraderAccountId = int(account_id_str)
            sub.symbolId.append(int(symbol_id))
            c.send(sub)
            print("Subscribed ✅ waiting for ticks...")
            return

        if isinstance(payload, ProtoOASpotEvent):
            bid_raw = float(getattr(payload, "bid", 0) or 0)
            ask_raw = float(getattr(payload, "ask", 0) or 0)

            # Auto-scale if values look like scaled integers
            if scale_override:
                bid = bid_raw / scale
                ask = ask_raw / scale
            else:
                bid = (bid_raw / scale) if bid_raw > 1000 else bid_raw
                ask = (ask_raw / scale) if ask_raw > 1000 else ask_raw

            # Repair zero ticks by carrying forward last known-good values
            if bid <= 0:
                if last_bid is None:
                    return
                bid = last_bid
            else:
                last_bid = bid

            if ask <= 0:
                if last_ask is None:
                    return
                ask = last_ask
            else:
                last_ask = ask

            # Optional write throttling
            now = time.time()
            if min_write_interval_sec > 0 and (now - last_write_ts) < min_write_interval_sec:
                # Still print, but don’t write every micro-tick if you don’t want to
                ts = now_utc_iso()
                label = symbol_name or str(symbol_id)
                if print_raw:
                    print(f"[{ts}] {label} bid={bid_raw:.2f} ask={ask_raw:.2f} (throttled)")
                else:
                    print(f"[{ts}] {label} bid={fmt.format(bid)} ask={fmt.format(ask)} (throttled)")
                return

            last_write_ts = now

            ts = now_utc_iso()

            db = connect()
            db.execute(
                "INSERT INTO quotes (ts_utc, env, account_id, symbol_id, bid, ask) VALUES (?,?,?,?,?,?)",
                (ts, env, str(account_id_str), int(symbol_id), float(bid), float(ask)),
            )
            db.commit()
            db.close()

            label = symbol_name or str(symbol_id)
            if print_raw:
                print(f"[{ts}] {label} bid={bid_raw:.2f} ask={ask_raw:.2f}")
            else:
                print(f"[{ts}] {label} bid={fmt.format(bid)} ask={fmt.format(ask)}")
            return

    c.setConnectedCallback(connected)
    c.setDisconnectedCallback(disconnected)
    c.setMessageReceivedCallback(on_message)

    c.startService()
    reactor.run()


if __name__ == "__main__":
    main()
