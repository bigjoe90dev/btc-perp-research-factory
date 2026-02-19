from __future__ import annotations

from research.engine.cv import walk_forward_splits


def test_walk_forward_respects_purge() -> None:
    splits = walk_forward_splits(
        n_bars=1000,
        train_bars=300,
        test_bars=100,
        step_bars=100,
        purge_bars=10,
        embargo_bars=5,
    )
    assert len(splits) > 0
    for sp in splits:
        assert sp.train_idx[-1] < sp.test_idx[0]
        assert (sp.test_idx[0] - sp.train_idx[-1]) >= 11
