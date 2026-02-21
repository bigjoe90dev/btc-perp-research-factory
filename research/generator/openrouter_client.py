from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMResponse:
    model_id: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(int(max_retries), 0)

    def chat_json(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code >= 500:
                    raise RuntimeError(f"OpenRouter 5xx: {resp.status_code} {resp.text[:500]}")
                if resp.status_code >= 400:
                    raise RuntimeError(f"OpenRouter error: {resp.status_code} {resp.text[:500]}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("OpenRouter response missing choices")
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("OpenRouter response missing message content")

                usage = data.get("usage") or {}
                p_tok = int(usage.get("prompt_tokens", 0) or 0)
                c_tok = int(usage.get("completion_tokens", 0) or 0)
                # Validate parseable JSON early for fail-closed behavior.
                json.loads(content)

                return LLMResponse(
                    model_id=model_id,
                    content=content,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    raw=data,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"OpenRouter call failed for model={model_id}: {last_err}")
