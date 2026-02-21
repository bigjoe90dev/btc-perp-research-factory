from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.engine.types import CandidateSpec

from .adapter import CandidateDraft, ideas_to_candidate_drafts
from .config import LLMGeneratorConfig, llm_config_from_backtest
from .data_summary import DataSummary, build_data_summary
from .openrouter_client import OpenRouterClient
from .prefilter import apply_prefilter
from .prompting import (
    build_improvement_prompt,
    build_synthesis_prompt,
    load_prompt_template,
    render_generation_prompt,
    sha256_text,
)
from .schema import ParsedIdea, parse_model_payload


@dataclass(frozen=True)
class GenerationResult:
    generation_id: str
    artefact_dir: Path
    candidate_file: Path
    candidates: list[CandidateSpec]
    manifest: dict[str, Any]


def _token_estimate(text: str) -> int:
    return max(int(math.ceil(len(text) / 4.0)), 1)


def _is_grok(model_id: str) -> bool:
    return "grok" in model_id.lower()


def _pricing_rates(cfg: LLMGeneratorConfig, model_id: str) -> tuple[float, float]:
    by_model = cfg.pricing.by_model.get(model_id, {}) if isinstance(cfg.pricing.by_model, dict) else {}
    in_rate = float(by_model.get("input_per_1k", cfg.pricing.default_input_per_1k))
    out_rate = float(by_model.get("output_per_1k", cfg.pricing.default_output_per_1k))
    return in_rate, out_rate


def _estimate_cost_usd(cfg: LLMGeneratorConfig, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = _pricing_rates(cfg=cfg, model_id=model_id)
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate


def _candidate_to_json(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "strategy_id": spec.strategy_id,
        "family": spec.family,
        "timeframe": spec.timeframe,
        "params": spec.params,
        "rules_version": spec.rules_version,
        "dataset_key": spec.dataset_key,
    }


def _draft_to_json(d: CandidateDraft) -> dict[str, Any]:
    row = _candidate_to_json(d.spec)
    row.update(
        {
            "name": d.idea.name,
            "description": d.idea.description,
            "confidence": d.idea.confidence,
            "expected_turnover": d.idea.expected_turnover,
            "why_it_passes_gates": d.idea.why_it_passes_gates,
            "source_model": d.idea.source_model,
            "source_lane": d.idea.source_lane,
        }
    )
    return row


def _generation_id(data_summary_json: str, cfg: LLMGeneratorConfig) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    key = json.dumps(
        {
            "summary": data_summary_json,
            "prompt_version": cfg.prompt_version,
            "lanes": [(x.name, x.model_id, x.calls) for x in cfg.lanes],
            "synthesis": cfg.synthesis_model_id,
        },
        sort_keys=True,
    )
    suffix = sha256_text(key)[:10]
    return f"gen_{stamp}_{suffix}"


def _write_generation_report(path: Path, selected: list[CandidateDraft], rejects: list[str]) -> None:
    lines = [
        "# Generation Report",
        "",
        f"Generated candidates: {len(selected)}",
        f"Rejected items: {len(rejects)}",
        "",
        "| Rank | Family | Name | Confidence | Expected Turnover | Why It Passes Gates | Suggested Timeframes |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for i, d in enumerate(selected, start=1):
        why = d.idea.why_it_passes_gates.replace("|", "/").replace("\n", " ").strip()
        why = why[:220] + ("..." if len(why) > 220 else "")
        lines.append(
            f"| {i} | {d.spec.family} | {d.idea.name} | {d.idea.confidence:.2f} | {d.idea.expected_turnover} | {why} | {','.join(d.idea.suggested_timeframes)} |"
        )

    if rejects:
        lines.extend(["", "## Rejections", ""])
        for r in rejects[:200]:
            lines.append(f"- `{r}`")
        if len(rejects) > 200:
            lines.append(f"- ... and {len(rejects) - 200} more")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_idea_rows(ideas: list[ParsedIdea]) -> list[dict[str, Any]]:
    return [
        {
            "family": i.family,
            "name": i.name,
            "description": i.description,
            "params": i.params,
            "expected_turnover": i.expected_turnover,
            "confidence": i.confidence,
            "why_it_passes_gates": i.why_it_passes_gates,
            "suggested_timeframes": i.suggested_timeframes,
            "source_model": i.source_model,
            "source_lane": i.source_lane,
        }
        for i in ideas
    ]


def _idea_fingerprint(idea: ParsedIdea) -> tuple[str, str]:
    return (idea.family.strip().lower(), idea.name.strip().lower())


def _restrict_ideas_to_source(
    candidate_ideas: list[ParsedIdea],
    source_ideas: list[ParsedIdea],
) -> tuple[list[ParsedIdea], list[str]]:
    allowed = {_idea_fingerprint(x) for x in source_ideas}
    kept: list[ParsedIdea] = []
    rejects: list[str] = []
    for idx, idea in enumerate(candidate_ideas):
        if _idea_fingerprint(idea) in allowed:
            kept.append(idea)
        else:
            rejects.append(f"synthesis_invented_strategy:{idx}:{idea.family}:{idea.name}")
    return kept, rejects


def _pricing_is_all_zero(cfg: LLMGeneratorConfig) -> bool:
    if float(cfg.pricing.default_input_per_1k) != 0.0 or float(cfg.pricing.default_output_per_1k) != 0.0:
        return False
    for rates in cfg.pricing.by_model.values():
        if not isinstance(rates, dict):
            continue
        if float(rates.get("input_per_1k", 0.0)) != 0.0 or float(rates.get("output_per_1k", 0.0)) != 0.0:
            return False
    return True


def generate_candidates_with_llm(
    data_config_path: str,
    backtest_cfg: dict[str, Any],
    count: int,
    timeframes: list[str],
    dataset_key: str,
    rules_version: str,
    out_path: str | None = None,
    estimate_only: bool = False,
) -> GenerationResult:
    cfg = llm_config_from_backtest(backtest_cfg)
    if not timeframes:
        raise ValueError("timeframes required")

    try:
        data_summary = build_data_summary(data_config_path=data_config_path, timeframe=str(timeframes[0]))
    except Exception as exc:  # noqa: BLE001
        if not estimate_only:
            raise
        # Allow estimate-only budgeting before local data is downloaded.
        data_summary = DataSummary(
            timeframe=str(timeframes[0]),
            summary={
                "timeframe": str(timeframes[0]),
                "status": "data_unavailable_for_estimate",
                "error": str(exc),
            },
        )
    data_summary_json = data_summary.as_json()

    template = load_prompt_template(cfg.prompt_path)
    prompt = render_generation_prompt(
        template=template,
        data_summary_json=data_summary_json,
        strategies_per_call=cfg.strategies_per_call,
    )
    prompt_hash = sha256_text(prompt)

    generation_id = _generation_id(data_summary_json=data_summary_json, cfg=cfg)
    artefact_dir = Path("research/artefacts/generation") / generation_id
    artefact_dir.mkdir(parents=True, exist_ok=True)

    render_prompt_path = artefact_dir / "rendered_prompt.md"
    render_prompt_path.write_text(prompt, encoding="utf-8")

    call_plan: list[tuple[str, str, str]] = []
    for lane in cfg.lanes:
        for _ in range(lane.calls):
            call_plan.append(("generate", lane.name, lane.model_id))

    planned_total_calls = len(call_plan) + 1 + (cfg.improvement_calls if cfg.improvement_round_enabled else 0)
    if planned_total_calls > cfg.budget.max_total_calls:
        raise RuntimeError(
            f"Planned calls exceed max_total_calls: planned={planned_total_calls}, max={cfg.budget.max_total_calls}"
        )

    prompt_tokens_est = _token_estimate(prompt)
    completion_tokens_est = min(cfg.budget.max_completion_tokens_per_call, 1200)

    estimated_calls = list(call_plan)
    estimated_calls.append(("synthesis", "grok_synth", cfg.synthesis_model_id))
    if cfg.improvement_round_enabled and cfg.improvement_calls > 0:
        for _ in range(cfg.improvement_calls):
            estimated_calls.append(("improve", "improve", cfg.improvement_model_id))

    estimated_cost = 0.0
    for _, _, model in estimated_calls:
        estimated_cost += _estimate_cost_usd(
            cfg=cfg,
            model_id=model,
            prompt_tokens=prompt_tokens_est,
            completion_tokens=completion_tokens_est,
        )
    warnings: list[str] = []
    if _pricing_is_all_zero(cfg):
        warnings.append(
            "pricing_all_zero: llm_generation.pricing values are all 0.0; cost reporting will be zero unless configured"
        )

    if estimate_only:
        manifest = {
            "generation_id": generation_id,
            "estimate_only": True,
            "prompt_version": cfg.prompt_version,
            "prompt_sha256": prompt_hash,
            "estimated_calls": len(estimated_calls),
            "estimated_prompt_tokens_per_call": prompt_tokens_est,
            "estimated_completion_tokens_per_call": completion_tokens_est,
            "estimated_total_cost_usd": round(estimated_cost, 6),
            "model_plan": estimated_calls,
            "warnings": warnings,
        }
        (artefact_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out = Path(out_path) if out_path else artefact_dir / "candidates_validated.json"
        out.write_text(json.dumps({"generation_id": generation_id, "candidates": []}, indent=2), encoding="utf-8")
        return GenerationResult(
            generation_id=generation_id,
            artefact_dir=artefact_dir,
            candidate_file=out,
            candidates=[],
            manifest=manifest,
        )

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    client = OpenRouterClient(
        api_key=api_key,
        base_url=cfg.openrouter_base_url,
        timeout_seconds=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )

    raw_outputs: list[dict[str, Any]] = []
    reject_reasons: list[str] = []
    parsed_ideas: list[ParsedIdea] = []
    total_completion_tokens = 0
    grok_generation_successes = 0

    for call_idx, (kind, lane_name, model_id) in enumerate(call_plan, start=1):
        resp = client.chat_json(
            model_id=model_id,
            system_prompt="You are a strict JSON-only BTC perp strategy generator.",
            user_prompt=prompt,
            max_tokens=cfg.budget.max_completion_tokens_per_call,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        total_completion_tokens += int(resp.completion_tokens)
        if total_completion_tokens > cfg.budget.max_total_completion_tokens:
            raise RuntimeError(
                f"Completion token cap exceeded: used={total_completion_tokens}, max={cfg.budget.max_total_completion_tokens}"
            )

        ideas, reasons = parse_model_payload(content=resp.content, model_id=model_id, lane_name=lane_name)
        parsed_ideas.extend(ideas)
        reject_reasons.extend([f"call_{call_idx}:{x}" for x in reasons])
        if _is_grok(model_id) and ideas:
            grok_generation_successes += 1

        raw_outputs.append(
            {
                "kind": kind,
                "lane": lane_name,
                "model_id": model_id,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "content": resp.content,
                "usage": resp.raw.get("usage", {}),
                "request_settings": {
                    "temperature": cfg.temperature,
                    "top_p": cfg.top_p,
                    "max_tokens": cfg.budget.max_completion_tokens_per_call,
                },
            }
        )

    if cfg.fail_closed_on_missing_grok and any(_is_grok(x.model_id) and x.calls > 0 for x in cfg.lanes):
        if grok_generation_successes == 0:
            raise RuntimeError("Grok generation lane produced no valid ideas; fail-closed policy active")

    if not parsed_ideas:
        raise RuntimeError("No valid strategy ideas parsed from generation round")

    raw_ideas_json = json.dumps({"strategies": _to_idea_rows(parsed_ideas)}, ensure_ascii=True)
    synthesis_prompt = build_synthesis_prompt(raw_ideas_json=raw_ideas_json, top_n=cfg.top_n_final)
    synth_resp = client.chat_json(
        model_id=cfg.synthesis_model_id,
        system_prompt="You are a strict JSON-only synthesis reviewer for BTC perp strategies.",
        user_prompt=synthesis_prompt,
        max_tokens=cfg.budget.max_completion_tokens_per_call,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    total_completion_tokens += int(synth_resp.completion_tokens)
    if total_completion_tokens > cfg.budget.max_total_completion_tokens:
        raise RuntimeError(
            f"Completion token cap exceeded: used={total_completion_tokens}, max={cfg.budget.max_total_completion_tokens}"
        )
    raw_outputs.append(
        {
            "kind": "synthesis",
            "lane": "grok_synth",
            "model_id": cfg.synthesis_model_id,
            "prompt_tokens": synth_resp.prompt_tokens,
            "completion_tokens": synth_resp.completion_tokens,
            "content": synth_resp.content,
            "usage": synth_resp.raw.get("usage", {}),
            "request_settings": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.budget.max_completion_tokens_per_call,
            },
        }
    )
    curated_ideas, synth_reasons = parse_model_payload(
        content=synth_resp.content,
        model_id=cfg.synthesis_model_id,
        lane_name="grok_synth",
    )
    reject_reasons.extend([f"synthesis:{x}" for x in synth_reasons])

    if cfg.fail_closed_on_missing_grok and not curated_ideas:
        raise RuntimeError(
            f"Synthesis model returned no valid ideas: model={cfg.synthesis_model_id}, fail-closed policy active"
        )
    if not curated_ideas:
        curated_ideas = parsed_ideas[: cfg.top_n_final]

    # Guard against synthesis-stage hallucinations by allowing only items that appeared
    # in the generation round.
    curated_ideas, source_rejects = _restrict_ideas_to_source(
        candidate_ideas=curated_ideas,
        source_ideas=parsed_ideas,
    )
    reject_reasons.extend(source_rejects)
    if cfg.fail_closed_on_missing_grok and not curated_ideas:
        raise RuntimeError(
            f"Synthesis model proposed no source-matching ideas: model={cfg.synthesis_model_id}, fail-closed policy active"
        )
    if not curated_ideas:
        curated_ideas = parsed_ideas[: cfg.top_n_final]

    if cfg.improvement_round_enabled and cfg.improvement_calls > 0:
        improve_input = json.dumps({"strategies": _to_idea_rows(curated_ideas)}, ensure_ascii=True)
        for _ in range(cfg.improvement_calls):
            prev_curated = curated_ideas
            improve_resp = client.chat_json(
                model_id=cfg.improvement_model_id,
                system_prompt="You are a strict JSON-only strategy improver.",
                user_prompt=build_improvement_prompt(curated_ideas_json=improve_input),
                max_tokens=cfg.budget.max_completion_tokens_per_call,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
            total_completion_tokens += int(improve_resp.completion_tokens)
            if total_completion_tokens > cfg.budget.max_total_completion_tokens:
                raise RuntimeError(
                    f"Completion token cap exceeded: used={total_completion_tokens}, max={cfg.budget.max_total_completion_tokens}"
                )
            raw_outputs.append(
                {
                    "kind": "improve",
                    "lane": "improve",
                    "model_id": cfg.improvement_model_id,
                    "prompt_tokens": improve_resp.prompt_tokens,
                    "completion_tokens": improve_resp.completion_tokens,
                    "content": improve_resp.content,
                    "usage": improve_resp.raw.get("usage", {}),
                    "request_settings": {
                        "temperature": cfg.temperature,
                        "top_p": cfg.top_p,
                        "max_tokens": cfg.budget.max_completion_tokens_per_call,
                    },
                }
            )
            improved, improve_reasons = parse_model_payload(
                content=improve_resp.content,
                model_id=cfg.improvement_model_id,
                lane_name="improve",
            )
            reject_reasons.extend([f"improve:{x}" for x in improve_reasons])
            if improved:
                improved_drafts, _ = ideas_to_candidate_drafts(
                    ideas=improved,
                    requested_timeframes=timeframes,
                    rules_version=rules_version,
                    dataset_key=dataset_key,
                )
                if improved_drafts:
                    curated_ideas = improved
                    improve_input = json.dumps({"strategies": _to_idea_rows(curated_ideas)}, ensure_ascii=True)
                else:
                    curated_ideas = prev_curated
                    reject_reasons.append("improve:adapter_empty_restore_previous")

    drafts, adapter_rejects = ideas_to_candidate_drafts(
        ideas=curated_ideas,
        requested_timeframes=timeframes,
        rules_version=rules_version,
        dataset_key=dataset_key,
    )
    reject_reasons.extend([f"adapter:{x}" for x in adapter_rejects])

    pre = apply_prefilter(
        drafts=drafts,
        max_count=max(int(count), 1),
        diversity_param_overlap_max=cfg.diversity_param_overlap_max,
    )
    selected_drafts = pre.selected
    reject_reasons.extend([f"prefilter:{x}" for x in pre.rejected])

    specs = [x.spec for x in selected_drafts]

    raw_out_path = artefact_dir / "raw_model_outputs.jsonl"
    with raw_out_path.open("w", encoding="utf-8") as fh:
        for row in raw_outputs:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    (artefact_dir / "candidates_raw.json").write_text(
        json.dumps({"strategies": _to_idea_rows(curated_ideas)}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    out_candidate_path = Path(out_path) if out_path else artefact_dir / "candidates_validated.json"
    out_candidate_path.parent.mkdir(parents=True, exist_ok=True)
    validated_payload = {
        "generation_id": generation_id,
        "candidates": [_candidate_to_json(s) for s in specs],
    }
    out_candidate_path.write_text(
        json.dumps(validated_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    # Keep an authoritative canonical copy in the artefact directory.
    artefact_validated_path = artefact_dir / "candidates_validated.json"
    if artefact_validated_path.resolve() != out_candidate_path.resolve():
        artefact_validated_path.write_text(
            json.dumps(validated_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    (artefact_dir / "rejections.json").write_text(
        json.dumps(reject_reasons, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    _write_generation_report(path=artefact_dir / "generation_report.md", selected=selected_drafts, rejects=reject_reasons)

    total_prompt_tokens = int(sum(int(x.get("prompt_tokens", 0) or 0) for x in raw_outputs))
    total_completion = int(sum(int(x.get("completion_tokens", 0) or 0) for x in raw_outputs))
    actual_cost = 0.0
    for row in raw_outputs:
        actual_cost += _estimate_cost_usd(
            cfg=cfg,
            model_id=str(row.get("model_id", "")),
            prompt_tokens=int(row.get("prompt_tokens", 0) or 0),
            completion_tokens=int(row.get("completion_tokens", 0) or 0),
        )

    manifest = {
        "generation_id": generation_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prompt_version": cfg.prompt_version,
        "prompt_sha256": prompt_hash,
        "rendered_prompt_path": str(render_prompt_path),
        "prompt_rendered": prompt,
        "data_summary": data_summary.summary,
        "model_plan": [(x[0], x[1], x[2]) for x in estimated_calls],
        "fail_closed_on_missing_grok": cfg.fail_closed_on_missing_grok,
        "usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion,
            "total_calls": len(raw_outputs),
            "estimated_total_cost_usd": round(estimated_cost, 6),
            "actual_total_cost_usd": round(actual_cost, 6),
        },
        "warnings": warnings,
        "outputs": {
            "candidate_count": len(specs),
            "raw_ideas_count": len(parsed_ideas),
            "curated_ideas_count": len(curated_ideas),
            "rejections_count": len(reject_reasons),
            "candidate_file": str(out_candidate_path),
        },
    }
    (artefact_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    (artefact_dir / "candidates_validated_enriched.json").write_text(
        json.dumps(
            {
                "generation_id": generation_id,
                "candidates": [_draft_to_json(d) for d in selected_drafts],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    return GenerationResult(
        generation_id=generation_id,
        artefact_dir=artefact_dir,
        candidate_file=out_candidate_path,
        candidates=specs,
        manifest=manifest,
    )
