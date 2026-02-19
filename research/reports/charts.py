from __future__ import annotations

from pathlib import Path
from typing import Any


def write_equity_snapshots(run_dir: Path, top_rows: list[dict[str, Any]]) -> list[str]:
    """
    Lightweight artifact writer without plotting dependencies.
    Stores top strategy equity curves as CSV.
    """
    out_paths: list[str] = []
    eq_dir = run_dir / "equity_curves"
    eq_dir.mkdir(parents=True, exist_ok=True)

    for row in top_rows:
        eq = row.get("equity_curve")
        sid = str(row.get("strategy_id"))
        if eq is None:
            continue
        p = eq_dir / f"{sid}.csv"
        eq.to_csv(p, index=False)
        out_paths.append(str(p))

    return out_paths
