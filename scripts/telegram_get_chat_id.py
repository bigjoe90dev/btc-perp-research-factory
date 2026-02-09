import requests
from shared.config import load_settings

def main():
    s = load_settings()

    print("1) In Telegram, open your bot and send: hi")
    print("2) Then run this script again if it says no updates.\n")

    url = f"https://api.telegram.org/bot{s.telegram_bot_token}/getUpdates"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = data.get("result", [])
    if not results:
        print("No updates found yet. Send the bot a message, then re-run.")
        return

    last = results[-1]
    chat_id = last["message"]["chat"]["id"]
    print(f"✅ Found chat_id: {chat_id}")
    print("Paste this into your .env as TELEGRAM_CHAT_ID=...")

if __name__ == "__main__":
    main()
