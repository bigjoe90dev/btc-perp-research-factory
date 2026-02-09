import json
from pathlib import Path
import requests

from shared.config import load_settings


TOKENS_PATHS = [Path("data/ctrader_tokens.json"), Path("ctrader_tokens.json")]


def load_tokens() -> dict:
    for p in TOKENS_PATHS:
        if p.exists():
            d = json.loads(p.read_text())
            return d
    raise RuntimeError("No ctrader_tokens.json found (expected in data/ctrader_tokens.json)")


def get_token_value(d: dict, *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def save_tokens(d: dict) -> Path:
    out = Path("data/ctrader_tokens.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2, sort_keys=True))
    return out


def main():
    s = load_settings()
    tokens = load_tokens()

    refresh = get_token_value(tokens, "refreshToken", "refresh_token")
    if not refresh:
        raise RuntimeError("ctrader_tokens.json missing refreshToken/refresh_token")

    url = "https://openapi.ctrader.com/apps/token"
    params = {
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": s.ctrader_client_id,
        "client_secret": s.ctrader_client_secret,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Open API returns errorCode=null on success :contentReference[oaicite:2]{index=2}
    if data.get("errorCode"):
        raise RuntimeError(f"Refresh failed: {data}")

    # Merge/update tokens (keep both naming styles if present)
    tokens.update(data)

    saved_to = save_tokens(tokens)
    print("OK: refreshed access token and saved ->", saved_to)


if __name__ == "__main__":
    main()
