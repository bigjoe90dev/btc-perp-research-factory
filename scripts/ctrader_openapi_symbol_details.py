import json
import os
from pathlib import Path

from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf

from shared.config import load_settings

# We import the pb2 module, then dynamically pick the correct message names
from ctrader_open_api.messages import OpenApiMessages_pb2 as M
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes


def _load_access_token(tokens_path: Path) -> str:
    d = json.loads(tokens_path.read_text())
    return (d.get("access_token") or d.get("accessToken") or "").strip()


def _pick(name_options):
    for n in name_options:
        if hasattr(M, n):
            return getattr(M, n)
    raise ImportError(f"None of these message types exist in your SDK: {name_options}")


def main():
    s = load_settings(require_ctrader=True)

    env = (os.getenv("CTRADER_ENV") or "demo").strip().lower()
    host = EndPoints.PROTOBUF_DEMO_HOST if env == "demo" else EndPoints.PROTOBUF_LIVE_HOST
    port = EndPoints.PROTOBUF_PORT

    account_id = int((os.getenv("CTRADER_ACCOUNT_ID") or "0").strip() or "0")
    symbol_id = int((os.getenv("CTRADER_SYMBOL_ID") or "0").strip() or "0")
    if not account_id or not symbol_id:
        raise RuntimeError("Missing CTRADER_ACCOUNT_ID / CTRADER_SYMBOL_ID in .env")

    client_id = (s.ctrader_client_id or "").strip()
    client_secret = (s.ctrader_client_secret or "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env")

    access_token = _load_access_token(s.tokens_path)
    if not access_token:
        raise RuntimeError(f"Missing access token in {s.tokens_path}. Re-run ctrader_oauth_full.py")

    # Pick message classes that exist in YOUR installed SDK
    AppAuthReq = _pick(["ProtoOAApplicationAuthReq"])
    AppAuthRes = _pick(["ProtoOAApplicationAuthRes"])

    AcctAuthReq = _pick(["ProtoOAAccountAuthReq"])
    AcctAuthRes = _pick(["ProtoOAAccountAuthRes"])

    # Symbols list message names vary between SDK builds
    SymbolsListReq = _pick(["ProtoOASymbolsListReq", "ProtoOAGetSymbolsListReq"])
    SymbolsListRes = _pick(["ProtoOASymbolsListRes", "ProtoOAGetSymbolsListRes"])

    c = Client(host, port, TcpProtocol)

    def stop(msg=""):
        if msg:
            print(msg)
        try:
            c.stopService()
        except Exception:
            pass
        try:
            reactor.stop()
        except Exception:
            pass

    def send_app_auth():
        req = AppAuthReq()
        req.clientId = str(client_id)
        req.clientSecret = str(client_secret)
        c.send(req)

    def send_account_auth():
        req = AcctAuthReq()
        req.ctidTraderAccountId = account_id
        req.accessToken = access_token
        c.send(req)

    def send_symbols_list():
        req = SymbolsListReq()
        # Some versions require account id on the request
        if hasattr(req, "ctidTraderAccountId"):
            req.ctidTraderAccountId = account_id
        c.send(req)

    def connected(_):
        print(f"Connected ✅ env={env}")
        send_app_auth()

    def disconnected(_, reason):
        # clean disconnect after stop is fine
        print("Disconnected:", reason)
        stop()

    def on_msg(_, message):
        payload = Protobuf.extract(message)

        if isinstance(payload, ProtoErrorRes):
            print("❌ OpenAPI Error:", getattr(payload, "errorCode", None), getattr(payload, "description", None))
            stop()
            return

        if isinstance(payload, AppAuthRes):
            print("App auth OK ✅")
            send_account_auth()
            return

        if isinstance(payload, AcctAuthRes):
            print("Account auth OK ✅")
            send_symbols_list()
            return

        if isinstance(payload, SymbolsListRes):
            # Field name varies: "symbol" vs "symbols"
            syms = []
            if hasattr(payload, "symbol"):
                syms = list(payload.symbol)
            elif hasattr(payload, "symbols"):
                syms = list(payload.symbols)

            if not syms:
                print("No symbols returned ❌")
                stop()
                return

            target = None
            for sym in syms:
                sid = getattr(sym, "symbolId", None)
                if sid == symbol_id:
                    target = sym
                    break

            if not target:
                print(f"SymbolId {symbol_id} not found in symbols list ❌")
                print("Tip: try CTRADER_SYMBOL_ID from your symbol search output.")
                stop()
                return

            name = getattr(target, "name", None)
            digits = getattr(target, "digits", None)
            pip_pos = getattr(target, "pipPosition", None)

            print("\nSymbol details ✅")
            print("symbolId:", symbol_id)
            print("name:", name)
            print("digits:", digits)
            print("pipPosition:", pip_pos)
            stop("Done ✅")
            return

    c.setConnectedCallback(connected)
    c.setDisconnectedCallback(disconnected)
    c.setMessageReceivedCallback(on_msg)

    c.startService()
    reactor.run()


if __name__ == "__main__":
    main()
