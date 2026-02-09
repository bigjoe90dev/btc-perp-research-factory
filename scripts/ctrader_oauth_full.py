#!/usr/bin/env python3
import json
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, quote

import requests

# Ensure local imports work when run as a script
# Usage: PYTHONPATH=. python scripts/ctrader_oauth_full.py
from shared.config import load_settings
from shared.telegram import send_telegram


@dataclass
class OAuthResult:
    code: str = ""
    error: str = ""


def build_auth_url(client_id: str, redirect_uri: str, scope: str) -> str:
    base = "https://connect.spotware.com/apps/auth"
    # client_id must include suffix, e.g. "12345_xxx"
    return (
        f"{base}?client_id={quote(client_id)}"
        f"&redirect_uri={quote(redirect_uri, safe='')}"
        f"&response_type=code"
        f"&scope={quote(scope)}"
    )


def exchange_code_for_tokens(client_id: str, client_secret: str, redirect_uri: str, code: str) -> dict:
    url = "https://connect.spotware.com/apps/token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }
    r = requests.post(url, data=payload, timeout=30)
    # Spotware often returns JSON even on non-200. Still parse it.
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Token endpoint did not return JSON. status={r.status_code}, body={r.text[:500]}")
    # Attach HTTP info for debugging without leaking secrets
    data["_http_status"] = r.status_code
    return data


def normalise_tokens(data: dict) -> dict:
    """
    Spotware sometimes returns camelCase keys and also snake_case duplicates.
    We'll normalise to snake_case for our storage.
    """
    err = data.get("errorCode")
    access = data.get("accessToken") or data.get("access_token")
    refresh = data.get("refreshToken") or data.get("refresh_token")
    expires = data.get("expiresIn") or data.get("expires_in")
    token_type = data.get("tokenType") or data.get("token_type") or "bearer"

    # Treat as failure ONLY if an actual error or missing tokens
    if err not in (None, "", "null") or not access or not refresh:
        # Let caller print full response
        return {
            "ok": False,
            "errorCode": err,
            "access_token": access,
            "refresh_token": refresh,
            "expires_in": expires,
            "token_type": token_type,
        }

    return {
        "ok": True,
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": int(expires) if expires is not None else None,
        "token_type": token_type,
    }


def run_callback_server(port: int, result: OAuthResult, stop_flag: dict):
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/callback":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            qs = parse_qs(parsed.query)
            code = (qs.get("code") or [""])[0]
            err = (qs.get("error") or [""])[0]

            if code:
                result.code = code
                body = b"Got code. You can close this tab. \xe2\x9c\x85"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            else:
                result.error = err or "No code in callback"
                body = b"No code received. You can close this tab. \xe2\x9d\x8c"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            # Tell the main thread we can stop
            stop_flag["stop"] = True

        def log_message(self, fmt, *args):
            # Keep logs tidy
            return

    httpd = HTTPServer(("127.0.0.1", port), CallbackHandler)

    # Serve until stop_flag flips or timeout hits
    start = time.time()
    while not stop_flag.get("stop", False):
        httpd.handle_request()
        if time.time() - start > 300:  # 5 minutes timeout
            break


def main():
    s = load_settings()

    if not s.ctrader_client_id or not s.ctrader_client_secret:
        raise RuntimeError("Missing CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET in .env")

    redirect_uri = s.ctrader_redirect_uri or "http://127.0.0.1:5555/callback"
    scope = s.ctrader_scope or "trading"
    port = 5555

    auth_url = build_auth_url(s.ctrader_client_id, redirect_uri, scope)

    print("\nOpen this URL in your browser, log in, and click Allow:\n")
    print(auth_url)
    print(f"\nWaiting for callback on {redirect_uri} ...\n")

    # Start callback server in a thread
    result = OAuthResult()
    stop_flag = {"stop": False}
    t = threading.Thread(target=run_callback_server, args=(port, result, stop_flag), daemon=True)
    t.start()

    # Wait until we have a code or error or timeout
    start = time.time()
    while not stop_flag.get("stop", False) and (time.time() - start) < 300:
        time.sleep(0.2)

    if result.error and not result.code:
        raise RuntimeError(f"OAuth callback error: {result.error}")

    if not result.code:
        raise RuntimeError("Timed out waiting for callback. Make sure the redirect URI matches and try again.")

    print("Got code ✅  Exchanging for tokens...\n")

    data = exchange_code_for_tokens(
        client_id=s.ctrader_client_id,
        client_secret=s.ctrader_client_secret,
        redirect_uri=redirect_uri,
        code=result.code,
    )

    norm = normalise_tokens(data)

    if not norm.get("ok"):
        print("Token exchange failed. Full response:\n")
        print(json.dumps(data, indent=2))
        raise RuntimeError("Token exchange failed (see response above). Usually client_id/secret/redirect mismatch.")

    # Save tokens
    out = {
        "access_token": norm["access_token"],
        "refresh_token": norm["refresh_token"],
        "expires_in": norm.get("expires_in"),
        "token_type": norm.get("token_type"),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/ctrader_tokens.json").write_text(json.dumps(out, indent=2))
    print("Saved tokens to data/ctrader_tokens.json ✅")

    # Optional: Telegram notify (won't crash the script if Telegram fails)
    try:
        send_telegram("✅ ict-bot: cTrader OAuth completed and tokens saved.")
    except Exception as e:
        print(f"(Telegram notify skipped: {e})")


if __name__ == "__main__":
    main()
