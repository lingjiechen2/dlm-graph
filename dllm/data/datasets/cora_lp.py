"""Cora link-prediction loader.

Thin wrapper around ``_lp_common.load_lp_split`` that plugs in the cora NC
loader. The cache path layout and returned dict shape are preserved exactly so
existing callers and on-disk caches keep working.
"""
from __future__ import annotations

from . import cora as cora_nc
from ._lp_common import load_lp_split


def load(
    config: dict,
    split: str,
    seed: int = 42,
    neg_ratio: int = 1,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
) -> dict:
    """Load the cora LP split. See ``_lp_common.load_lp_split`` for the
    returned dict schema."""
    return load_lp_split(
        dataset_name="cora",
        nc_loader=cora_nc.load,
        config=config,
        split=split,
        seed=seed,
        neg_ratio=neg_ratio,
        val_frac=val_frac,
        test_frac=test_frac,
    )
