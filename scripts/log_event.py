from datetime import datetime, timezone
from shared.db import connect

def log_event(level: str, source: str, message: str) -> None:
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO event_log (ts_utc, level, message) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), level, f"[{source}] {message}"),
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    log_event("INFO", "bootstrap", "DB + dashboard + telegram are live")
    print("Logged.")
