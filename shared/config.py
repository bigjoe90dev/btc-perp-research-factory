# shared/config.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _project_root() -> Path:
    # shared/ lives one level under repo root
    return Path(__file__).resolve().parents[1]


def _as_bool(v: str, default: bool = False) -> bool:
    if v is None:
        return default
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    # Telegram (optional; scripts/daemons can still run without it)
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str

    # cTrader OAuth (Connect / token refresh scripts)
    ctrader_client_id: str
    ctrader_client_secret: str
    ctrader_redirect_uri: str
    ctrader_scope: str

    # cTrader runtime config (so scripts don't prompt)
    ctrader_env: str                 # "demo" or "live"
    ctrader_account_id: str          # cTID Trader Account ID (e.g. 45922689)
    ctrader_symbol_id: int           # e.g. 1
    ctrader_symbol_name: str         # e.g. "EURUSD"
    ctrader_digits: int              # e.g. 5

    # Files
    tokens_path: Path
    db_path: Path


def load_settings(
    require_ctrader: bool = False,
    require_market: bool = False,
    require_telegram: bool = False,
) -> Settings:
    root = _project_root()

    # Load .env from repo root regardless of current working dir
    env_path = root / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    # Telegram
    telegram_enabled = _as_bool(os.getenv("TELEGRAM_ENABLED", "1"), default=True)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if (require_telegram or telegram_enabled) and (not bot_token or not chat_id):
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env (or set TELEGRAM_ENABLED=0)")

    # cTrader OAuth app creds
    ctrader_client_id = os.getenv("CTRADER_CLIENT_ID", "").strip()
    ctrader_client_secret = os.getenv("CTRADER_CLIENT_SECRET", "").strip()
    ctrader_redirect_uri = os.getenv("CTRADER_REDIRECT_URI", "http://127.0.0.1:5555/callback").strip()
    ctrader_scope = os.getenv("CTRADER_SCOPE", "trading").strip()

    if require_ctrader:
        if not ctrader_client_id:
            raise RuntimeError("Missing CTRADER_CLIENT_ID in .env")
        if not ctrader_client_secret:
            raise RuntimeError("Missing CTRADER_CLIENT_SECRET in .env")

    # Runtime / market config (so we can stop prompting)
    ctrader_env = (os.getenv("CTRADER_ENV", "demo").strip().lower() or "demo")
    if ctrader_env not in ("demo", "live"):
        raise RuntimeError("CTRADER_ENV must be 'demo' or 'live'")

    ctrader_account_id = os.getenv("CTRADER_ACCOUNT_ID", "").strip()
    ctrader_symbol_name = os.getenv("CTRADER_SYMBOL_NAME", "").strip() or "EURUSD"

    symbol_id_raw = os.getenv("CTRADER_SYMBOL_ID", "").strip()
    ctrader_symbol_id = int(symbol_id_raw) if symbol_id_raw else 0

    digits_raw = os.getenv("CTRADER_DIGITS", "").strip()
    ctrader_digits = int(digits_raw) if digits_raw else 5

    if require_market:
        if not ctrader_account_id or not ctrader_symbol_id:
            raise RuntimeError("Missing CTRADER_ACCOUNT_ID / CTRADER_SYMBOL_ID in .env")

    # Token file location
    tokens_path = root / "data" / "ctrader_tokens.json"
    tokens_path_override = os.getenv("CTRADER_TOKENS_PATH", "").strip()
    if tokens_path_override:
        tokens_path = Path(tokens_path_override).expanduser().resolve()

    # DB path (shared.db should already be using something like this)
    db_path = root / "data" / "ict_bot.sqlite3"
    db_path_override = os.getenv("ICT_DB_PATH", "").strip()
    if db_path_override:
        db_path = Path(db_path_override).expanduser().resolve()

    return Settings(
        telegram_enabled=telegram_enabled,
        telegram_bot_token=bot_token,
        telegram_chat_id=chat_id,
        ctrader_client_id=ctrader_client_id,
        ctrader_client_secret=ctrader_client_secret,
        ctrader_redirect_uri=ctrader_redirect_uri,
        ctrader_scope=ctrader_scope,
        ctrader_env=ctrader_env,
        ctrader_account_id=ctrader_account_id,
        ctrader_symbol_id=ctrader_symbol_id,
        ctrader_symbol_name=ctrader_symbol_name,
        ctrader_digits=ctrader_digits,
        tokens_path=tokens_path,
        db_path=db_path,
    )
