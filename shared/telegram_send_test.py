from shared.config import load_settings
from shared.telegram import send_telegram

if __name__ == "__main__":
    s = load_settings()
    send_telegram(s, "ict-bot ✅ Telegram is wired up and ready.")
    print("Sent.")
