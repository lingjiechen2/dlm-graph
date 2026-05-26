"""ogbn-arxiv link-prediction loader.

Thin wrapper around ``_lp_common.load_lp_split`` that plugs in the
ogbn-arxiv NC loader. Edges are split randomly (default 85/5/10) — note this
deliberately differs from OGB's official year-based node split, which only
applies to NC.
"""
from __future__ import annotations

from . import ogbn_arxiv as arxiv_nc
from ._lp_common import load_lp_llaga_split, load_lp_split


def load(
    config: dict,
    split: str,
    seed: int = 42,
    neg_ratio: int = 1,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
    use_llaga_split: bool = False,
) -> dict:
    """Load the ogbn-arxiv LP split. See ``_lp_common.load_lp_split`` for the
    returned dict schema. Set ``use_llaga_split=True`` to use LLaGA's official
    train/test JSONL instead of our random seed-42 split."""
    if use_llaga_split:
        return load_lp_llaga_split(
            dataset_name="ogbn-arxiv",
            nc_loader=arxiv_nc.load,
            config=config,
            split=split,
        )
    return load_lp_split(
        dataset_name="ogbn-arxiv",
        nc_loader=arxiv_nc.load,
        config=config,
        split=split,
        seed=seed,
        neg_ratio=neg_ratio,
        val_frac=val_frac,
        test_frac=test_frac,
    )
