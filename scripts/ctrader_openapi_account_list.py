import json
import sys
from pathlib import Path

from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol, EndPoints, Protobuf

from shared.config import load_settings

from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAGetAccountListByAccessTokenRes,
)
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes


def _load_access_token(tokens_path: Path) -> str:
    d = json.loads(tokens_path.read_text())
    token = d.get("access_token") or d.get("accessToken") or ""
    return (token or "").strip()


def main():
    settings = load_settings(require_ctrader=False)

    env = (input("Host (demo/live) [demo]: ").strip().lower() or "demo")
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
        raise RuntimeError(f"{settings.tokens_path} missing access token. Re-run scripts/ctrader_oauth_full.py")

    client = Client(host, port, TcpProtocol)

    finished = {"done": False}

    def stop(msg: str = ""):
        if finished["done"]:
            return
        finished["done"] = True

        if msg:
            print(msg)

        try:
            client.stopService()
        except Exception:
            pass

        if reactor.running:
            try:
                reactor.stop()
            except Exception:
                pass

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

    def connected(_client):
        print("Connected to Open API ✅")
        send_app_auth()

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
            print("App auth OK ✅")
            send_account_list()
            return

        if isinstance(payload, ProtoOAGetAccountListByAccessTokenRes):
            accounts = list(getattr(payload, "ctidTraderAccount", []))

            if not accounts:
                stop("\nNo accounts returned. (Wrong env? Try 'live' vs 'demo'.)\n")
                return

            print("\nAccounts found ✅")
            for a in accounts:
                acc_id = getattr(a, "ctidTraderAccountId", None)
                is_live = getattr(a, "isLive", None)
                broker = getattr(a, "brokerName", None) or getattr(a, "broker", None)
                name = getattr(a, "traderLogin", None) or getattr(a, "accountName", None)
                print(f"- accountId={acc_id}  isLive={is_live}  broker={broker}  name={name}")

            stop("\nDone ✅")
            return

    def disconnected(_client, reason):
        # This will often fire after stopService() or if the server closes the socket.
        # It's not an error in our case — keep it quiet once we've finished.
        if not finished["done"]:
            print(f"Disconnected: {reason}")
            stop()

    def on_timeout():
        if not finished["done"]:
            stop("\nTimed out waiting for account list ❌\n")

    client.setConnectedCallback(connected)
    client.setDisconnectedCallback(disconnected)
    client.setMessageReceivedCallback(on_message_received)

    reactor.callLater(15, on_timeout)

    client.startService()
    reactor.run()


if __name__ == "__main__":
    main()
