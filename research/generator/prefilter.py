from __future__ import annotations

from dataclasses import dataclass

from .adapter import CandidateDraft


@dataclass(frozen=True)
class PrefilterResult:
    selected: list[CandidateDraft]
    rejected: list[str]


def _param_overlap_ratio(a: dict, b: dict) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    if not keys:
        return 0.0
    same = 0
    for k in keys:
        if a.get(k) == b.get(k):
            same += 1
    return float(same / len(keys))


def apply_prefilter(
    drafts: list[CandidateDraft],
    max_count: int,
    diversity_param_overlap_max: float,
) -> PrefilterResult:
    rejects: list[str] = []

    by_id: dict[str, CandidateDraft] = {}
    for d in drafts:
        sid = d.spec.strategy_id
        if sid in by_id:
            rejects.append(f"duplicate_strategy_id:{sid}")
            continue
        by_id[sid] = d

    uniq = list(by_id.values())
    uniq.sort(
        key=lambda x: (
            float(x.idea.confidence),
            1.0 if x.idea.expected_turnover in {"low", "medium"} else 0.0,
            x.spec.strategy_id,
        ),
        reverse=True,
    )

    selected: list[CandidateDraft] = []
    for draft in uniq:
        too_similar = False
        for keep in selected:
            if draft.spec.family != keep.spec.family or draft.spec.timeframe != keep.spec.timeframe:
                continue
            overlap = _param_overlap_ratio(draft.spec.params, keep.spec.params)
            if overlap > diversity_param_overlap_max:
                rejects.append(
                    f"high_param_overlap:{draft.spec.strategy_id}:with={keep.spec.strategy_id}:overlap={overlap:.2f}"
                )
                too_similar = True
                break
        if too_similar:
            continue

        selected.append(draft)
        if len(selected) >= max_count:
            break

    return PrefilterResult(selected=selected, rejected=rejects)
