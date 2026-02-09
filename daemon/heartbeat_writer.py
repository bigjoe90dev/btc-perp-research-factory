import time
from datetime import datetime, timezone

from shared.config import load_settings
from shared.db import connect
from shared.telegram import send_telegram


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_heartbeat(status: str, note: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO heartbeat (ts_utc, status, note) VALUES (?, ?, ?)",
            (utc_now_iso(), status, note),
        )
        conn.commit()


def main() -> None:
    s = load_settings()
    send_telegram(s, "ict-bot: heartbeat daemon started ✅")

    while True:
        write_heartbeat("RUNNING", "heartbeat ok")
        time.sleep(5)


if __name__ == "__main__":
    main()
