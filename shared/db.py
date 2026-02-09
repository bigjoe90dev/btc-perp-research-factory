import os
import sqlite3
from pathlib import Path

DB_PATH_DEFAULT = Path("data/ict_bot.sqlite3")

def get_db_path() -> Path:
    p = os.getenv("DB_PATH", "").strip()
    return Path(p) if p else DB_PATH_DEFAULT

def connect() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn
