from __future__ import annotations

import hashlib
from pathlib import Path


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8")


def render_generation_prompt(template: str, data_summary_json: str, strategies_per_call: int) -> str:
    prompt = template.replace("{data_summary_json_block}", data_summary_json)
    # Keep control over call size even if template has a fixed number.
    prompt += (
        "\n\nAdditional runtime constraint for this call: "
        f"return at most {int(strategies_per_call)} strategies."
    )
    return prompt


def build_synthesis_prompt(raw_ideas_json: str, top_n: int) -> str:
    return (
        "You are the synthesis judge.\n"
        "Input contains candidate ideas from multiple models.\n"
        "Select the highest-quality, diverse strategies likely to survive strict BTC perp gates.\n"
        "Output JSON with key 'strategies' only.\n"
        f"Return exactly {int(top_n)} strategies when possible.\n\n"
        "INPUT:\n"
        f"{raw_ideas_json}"
    )


def build_improvement_prompt(curated_ideas_json: str) -> str:
    return (
        "Improve the provided strategies for lower turnover and stronger gate survival while preserving core logic.\n"
        "Output JSON with key 'strategies'.\n\n"
        "INPUT:\n"
        f"{curated_ideas_json}"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
