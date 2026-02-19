from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_bars: int,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    purge_bars: int,
    embargo_bars: int,
) -> list[Split]:
    if min(n_bars, train_bars, test_bars, step_bars) <= 0:
        return []

    splits: list[Split] = []
    cursor = 0
    fold = 0
    while True:
        train_start = cursor
        train_end = train_start + train_bars
        test_start = train_end + purge_bars
        test_end = test_start + test_bars
        if test_end > n_bars:
            break

        train_idx = np.arange(train_start, train_end, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        splits.append(Split(fold_id=fold, train_idx=train_idx, test_idx=test_idx))
        fold += 1

        cursor = cursor + step_bars + embargo_bars

    return splits


def purged_kfold_splits(
    n_bars: int,
    n_splits: int,
    purge_bars: int,
    embargo_bars: int,
) -> list[Split]:
    if n_splits <= 1 or n_bars <= 0:
        return []

    edges = np.linspace(0, n_bars, n_splits + 1, dtype=int)
    all_idx = np.arange(n_bars, dtype=int)

    out: list[Split] = []
    for fold, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        test_idx = np.arange(a, b, dtype=int)

        left_cut = max(0, a - purge_bars)
        right_cut = min(n_bars, b + embargo_bars)
        mask = (all_idx < left_cut) | (all_idx >= right_cut)
        train_idx = all_idx[mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        out.append(Split(fold_id=fold, train_idx=train_idx, test_idx=test_idx))

    return out
