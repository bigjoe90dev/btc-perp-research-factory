from shared.config import Settings, load_settings
import requests

def send_telegram(settings_or_text, maybe_text=None):
    """
    Supports:
      send_telegram(settings, "hi")
      send_telegram("hi")   # loads settings automatically
    """
    if maybe_text is None:
        settings = load_settings()
        text = str(settings_or_text)
    else:
        settings = settings_or_text
        text = str(maybe_text)

    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": settings.telegram_chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return True
