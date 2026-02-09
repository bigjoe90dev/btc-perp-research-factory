import json
import os
import urllib.parse
from pathlib import Path

import requests
from shared.config import load_settings

TOKENS_PATH = Path("data/ctrader_tokens.json")

def main():
    s = load_settings()

    code = os.getenv("CTRADER_AUTH_CODE", "").strip()
    if not code:
        code = input("Paste the cTrader ?code= value (full string): ").strip()

    params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": s.ctrader_redirect_uri,
        "client_id": s.ctrader_client_id,
        "client_secret": s.ctrader_client_secret,
    }

    url = "https://openapi.ctrader.com/apps/token?" + urllib.parse.urlencode(params)
    r = requests.get(url, headers={"Accept": "application/json"})

    data = r.json()
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.write_text(json.dumps(data, indent=2))
    print("Saved:", TOKENS_PATH)

    print("Top-level keys:", sorted(data.keys()))
    if data.get("errorCode"):
        print("ERROR:", data.get("errorCode"), "-", data.get("description"))
        raise SystemExit(1)

    if not data.get("accessToken"):
        raise SystemExit("No accessToken returned (unexpected).")

    print("OK: accessToken saved.")

if __name__ == "__main__":
    main()
