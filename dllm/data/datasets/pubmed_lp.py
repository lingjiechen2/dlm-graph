"""PubMed link-prediction loader.

Thin wrapper around ``_lp_common`` that plugs in the PubMed NC loader. The
LLaGA-split path is the one used for head-to-head LP experiments.
"""
from __future__ import annotations

from . import pubmed as pubmed_nc
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
    """Load the PubMed LP split. Set ``use_llaga_split=True`` to use LLaGA's
    official train/test JSONL instead of a random seed split."""
    if use_llaga_split:
        return load_lp_llaga_split(
            dataset_name="pubmed",
            nc_loader=pubmed_nc.load,
            config=config,
            split=split,
        )
    return load_lp_split(
        dataset_name="pubmed",
        nc_loader=pubmed_nc.load,
        config=config,
        split=split,
        seed=seed,
        neg_ratio=neg_ratio,
        val_frac=val_frac,
        test_frac=test_frac,
    )
