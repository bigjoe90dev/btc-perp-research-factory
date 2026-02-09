import os
import urllib.parse
from shared.config import load_settings

def main():
    s = load_settings()

    if not s.ctrader_client_id or not s.ctrader_redirect_uri:
        raise RuntimeError("Missing CTRADER_CLIENT_ID or CTRADER_REDIRECT_URI in .env")

    # cTrader uses OAuth2; scope depends on your app permissions
    scope = s.ctrader_scope or "trading"
    state = "ict-bot"

    params = {
        "response_type": "code",
        "client_id": s.ctrader_client_id,
        "redirect_uri": s.ctrader_redirect_uri,
        "scope": scope,
        "state": state,
    }

    base = "https://connect.spotware.com/apps/auth"
    url = base + "?" + urllib.parse.urlencode(params)

    print("\nOpen this URL in your browser:\n")
    print(url)
    print("\nAfter approving, you'll be redirected to the callback server.\n")

if __name__ == "__main__":
    main()
