import json
from pathlib import Path
from datetime import datetime, timezone

from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf

from shared.config import load_settings

import ctrader_open_api.messages.OpenApiMessages_pb2 as OA
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes


def load_access_token(tokens_path: Path) -> str:
    d = json.loads(tokens_path.read_text())
    token = d.get("access_token") or d.get("accessToken") or ""
    return (token or "").strip()


def rel_price_to_float(x: int) -> float:
    # cTrader Open API “relative” price: divide by 100000
    return float(x) / 100000.0


def ts_ms_to_iso(ts_ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return str(ts_ms)


def main():
    settings = load_settings(require_ctrader=False)

    # Inputs (we’ll automate later; for now you can just press Enter for defaults)
    env = (input("Host (demo/live) [demo]: ").strip().lower() or "demo")
    if env not in ("demo", "live"):
        raise SystemExit("Please type demo or live.")

    default_account_id = "45922689"
    default_symbol_id = "1"

    account_id = int(input(f"AccountId [{default_account_id}]: ").strip() or default_account_id)
    symbol_id = int(input(f"SymbolId (EURUSD=1) [{default_symbol_id}]: ").strip() or default_symbol_id)

    host = EndPoints.PROTOBUF_DEMO_HOST if env == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT

    client_id = (settings.ctrader_client_id or "").strip()
    client_secret = (settings.ctrader_client_secret or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env")

    access_token = load_access_token(settings.tokens_path)
    if not access_token:
        raise RuntimeError(f"Missing access token in {settings.tokens_path}. Re-run ctrader_oauth_full.py")

    client = Client(host, port, TcpProtocol)

    state = {
        "app_ok": False,
        "acct_ok": False,
        "sub_ok": False,
        "events": 0,
    }

    def stop(msg: str = ""):
        if msg:
            print(msg)
        try:
            client.stopService()
        except Exception:
            pass
        try:
            reactor.stop()
        except Exception:
            pass

    def send_app_auth():
        req = OA.ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        client.send(req)

    def send_account_auth():
        req = OA.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = account_id
        req.accessToken = access_token
        client.send(req)

    def send_subscribe_spots():
        req = OA.ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = account_id
        req.symbolId.append(symbol_id)
        req.subscribeToSpotTimestamp = True
        client.send(req)

    def connected(_client):
        print("Connected to Open API ✅")
        send_app_auth()

    def disconnected(_client, reason):
        # Don’t treat this as “failure” for the test; just exit cleanly.
        print(f"Disconnected: {reason}")
        stop()

    def on_message(_client, message):
        try:
            payload = Protobuf.extract(message)
        except Exception as e:
            print(f"⚠️ Could not decode message: {e}")
            return

        if isinstance(payload, ProtoErrorRes):
            print("\n❌ OpenAPI Error:")
            print(f"  errorCode={getattr(payload, 'errorCode', None)}")
            print(f"  description={getattr(payload, 'description', None)}\n")
            stop()
            return

        if isinstance(payload, OA.ProtoOAApplicationAuthRes):
            state["app_ok"] = True
            print("App auth OK ✅")
            send_account_auth()
            return

        if isinstance(payload, OA.ProtoOAAccountAuthRes):
            state["acct_ok"] = True
            print("Account auth OK ✅")
            send_subscribe_spots()
            return

        if isinstance(payload, OA.ProtoOASubscribeSpotsRes):
            state["sub_ok"] = True
            print(f"\nSubscribed ✅ accountId={account_id} symbolId={symbol_id}")
            print("Waiting for spot events… (printing for 120s)\n")
            return

        # This is the one we care about
        if isinstance(payload, OA.ProtoOASpotEvent):
            state["events"] += 1
            bid = getattr(payload, "bid", None)
            ask = getattr(payload, "ask", None)
            ts = getattr(payload, "timestamp", None) or getattr(payload, "spotTimestamp", None)
            sid = getattr(payload, "symbolId", None)

            if bid is not None and ask is not None:
                bid_f = rel_price_to_float(int(bid))
                ask_f = rel_price_to_float(int(ask))
                ts_s = ts_ms_to_iso(int(ts)) if ts is not None else "n/a"
                print(f"[{ts_s}] symbolId={sid} bid={bid_f:.5f} ask={ask_f:.5f}")
            else:
                print(f"SpotEvent received (missing bid/ask?) symbolId={sid}")
            return

        # Ignore everything else (heartbeats etc.)
        return

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message)

    client.startService()

    # Stop after 120 seconds so you don’t have to Ctrl+C
    reactor.callLater(120, lambda: stop("\nDone ✅"))
    reactor.run()


if __name__ == "__main__":
    main()
