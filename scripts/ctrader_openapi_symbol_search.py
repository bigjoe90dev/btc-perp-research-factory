import json
import sys
from pathlib import Path

from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf

from shared.config import load_settings

from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAGetAccountListByAccessTokenRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
)
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes


def _load_access_token(tokens_path: Path) -> str:
    d = json.loads(tokens_path.read_text())
    token = d.get("access_token") or d.get("accessToken") or ""
    return (token or "").strip()


def main():
    settings = load_settings(require_ctrader=False)

    env = input("Host (demo/live) [demo]: ").strip().lower() or "demo"
    if env not in ("demo", "live"):
        print("Please type 'demo' or 'live'.")
        sys.exit(1)

    host = EndPoints.PROTOBUF_DEMO_HOST if env == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT

    client_id = (settings.ctrader_client_id or "").strip()
    client_secret = (settings.ctrader_client_secret or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in your .env")

    access_token = _load_access_token(settings.tokens_path)
    if not access_token:
        raise RuntimeError(f"{settings.tokens_path} missing access token. Re-run ctrader_oauth_full.py")

    query = input("Symbol search (e.g. EURUSD, XAU, NAS): ").strip()
    if not query:
        query = "EUR"

    client = Client(host, port, TcpProtocol)

    state = {
        "app_authed": False,
        "account_id": None,
        "account_authed": False,
        "done": False,
        "timeout_call": None,
    }

    def stop(msg: str = ""):
        if state["done"]:
            return
        state["done"] = True
        if msg:
            print(msg)
        try:
            if state["timeout_call"] is not None and state["timeout_call"].active():
                state["timeout_call"].cancel()
        except Exception:
            pass
        try:
            client.stopService()
        except Exception:
            pass
        try:
            reactor.stop()
        except Exception:
            pass

    def arm_timeout(seconds: int, label: str):
        def _fire():
            stop(f"\nTimed out waiting for {label} ❌\n")
        state["timeout_call"] = reactor.callLater(seconds, _fire)

    def on_error_res(err: ProtoErrorRes):
        print("\n❌ OpenAPI Error:")
        print(f"  errorCode={getattr(err, 'errorCode', None)}")
        print(f"  description={getattr(err, 'description', None)}\n")
        stop()

    def send_app_auth():
        req = ProtoOAApplicationAuthReq()
        req.clientId = client_id
        req.clientSecret = client_secret
        client.send(req)

    def send_account_list():
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = access_token
        client.send(req)

    def send_account_auth(account_id: int):
        # Required before account-scoped requests like SymbolsList
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = int(account_id)
        req.accessToken = access_token
        client.send(req)

    def send_symbols_list(account_id: int):
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = int(account_id)
        client.send(req)

    def connected(_client):
        print("Connected to Open API ✅")
        arm_timeout(12, "App auth")
        send_app_auth()

    def disconnected(_client, reason):
        # normal when we stop the reactor after success
        if not state["done"]:
            print(f"Disconnected: {reason}")
        stop()

    def on_message_received(_client, message):
        try:
            payload = Protobuf.extract(message)
        except Exception as e:
            print(f"⚠️ Could not decode message: {e}")
            return

        if isinstance(payload, ProtoErrorRes):
            on_error_res(payload)
            return

        if isinstance(payload, ProtoOAApplicationAuthRes):
            state["app_authed"] = True
            print("App auth OK ✅")
            arm_timeout(12, "Account list")
            send_account_list()
            return

        if isinstance(payload, ProtoOAGetAccountListByAccessTokenRes):
            accounts = list(getattr(payload, "ctidTraderAccount", []))
            if not accounts:
                stop("No accounts returned ❌ (wrong demo/live or token scope?)")
                return

            # pick the first account for now (we can add selection later)
            a0 = accounts[0]
            account_id = int(getattr(a0, "ctidTraderAccountId"))
            state["account_id"] = account_id
            print(f"\nUsing accountId={account_id} ✅")

            arm_timeout(12, "Account auth")
            send_account_auth(account_id)
            return

        if isinstance(payload, ProtoOAAccountAuthRes):
            state["account_authed"] = True
            print("Account auth OK ✅")
            arm_timeout(20, "Symbols list")
            send_symbols_list(state["account_id"])
            return

        if isinstance(payload, ProtoOASymbolsListRes):
            symbols = list(getattr(payload, "symbol", []))
            if not symbols:
                stop("Symbols list returned empty ❌")
                return

            q = query.lower()
            matches = []
            for s in symbols:
                name = (getattr(s, "symbolName", "") or "").strip()
                if q in name.lower():
                    matches.append((getattr(s, "symbolId", None), name))

            print(f"\nSymbols matching '{query}' ✅  (showing up to 40)\n")
            for sid, name in matches[:40]:
                print(f"- symbolId={sid}  name={name}")

            if not matches:
                print("(No matches found.)")

            stop("\nDone ✅")
            return

        # ignore everything else (heartbeats etc.)
        return

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)

    client.startService()
    reactor.run()


if __name__ == "__main__":
    main()
