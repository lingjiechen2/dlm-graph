"""Per-dataset / per-class resampling helpers for TAG dataset merging.

Operates on per-dataset sample lists *after* tokenized samples are built.
Strategies:
  - "none": no change
  - "balance_datasets": downsample each dataset to min(per-dataset count)
  - "balance_classes": within each dataset, up/down-sample each class to the
                       dataset's class-median count
  - "boost": multiply specific (dataset, class) sample lists by an integer
             factor; spec is a comma-separated list "ds:class_name:factor"

Each sample dict is expected to carry "dataset" and "cls_label". The returned
dict has the same keys; sample lists may be longer (oversampling) or shorter
(downsampling) than the input.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Dict, List


def _parse_boost_spec(boost_spec: str) -> Dict[str, Dict[str, int]]:
    """Parse 'cora:Theory:2,cora:Rule Learning:3' -> {'cora': {'Theory': 2, 'Rule Learning': 3}}."""
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    if not boost_spec:
        return out
    for entry in boost_spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"boost_spec entry must be 'dataset:class_name:factor', got: {entry!r}"
            )
        ds, cls, factor = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            factor_int = int(factor)
        except ValueError as e:
            raise ValueError(f"boost factor must be int, got {factor!r}") from e
        if factor_int < 1:
            raise ValueError(f"boost factor must be >= 1, got {factor_int}")
        out[ds][cls] = factor_int
    return out


def _balance_datasets(
    per_dataset_samples: Dict[str, List[dict]],
    rng: random.Random,
) -> Dict[str, List[dict]]:
    if len(per_dataset_samples) <= 1:
        return per_dataset_samples
    target = min(len(s) for s in per_dataset_samples.values())
    out = {}
    for n, pool in per_dataset_samples.items():
        if len(pool) > target:
            out[n] = rng.sample(pool, target)
        else:
            out[n] = list(pool)
    return out


def _balance_classes(
    per_dataset_samples: Dict[str, List[dict]],
    rng: random.Random,
) -> Dict[str, List[dict]]:
    """Within each dataset, resample each class to that dataset's class-median count."""
    out: Dict[str, List[dict]] = {}
    for n, pool in per_dataset_samples.items():
        by_class: Dict[int, List[dict]] = defaultdict(list)
        for s in pool:
            by_class[s["cls_label"]].append(s)
        target = int(statistics.median([len(v) for v in by_class.values()]))
        target = max(target, 1)
        new_pool: List[dict] = []
        for cls, items in by_class.items():
            if len(items) >= target:
                new_pool.extend(rng.sample(items, target))
            else:
                # Oversample by sampling with replacement
                extra = target - len(items)
                new_pool.extend(items)
                new_pool.extend(rng.choices(items, k=extra))
        out[n] = new_pool
    return out


def _boost_classes(
    per_dataset_samples: Dict[str, List[dict]],
    boost_spec: str,
    class_names_per_dataset: Dict[str, List[str]],
    rng: random.Random,
) -> Dict[str, List[dict]]:
    """Multiply (dataset, class) sample lists by integer factor.

    Boosted samples are appended (each kept once + (factor-1) extra copies via
    sampling with replacement, so noise from random masking still varies them).
    """
    spec = _parse_boost_spec(boost_spec)
    out: Dict[str, List[dict]] = {}
    for n, pool in per_dataset_samples.items():
        ds_spec = spec.get(n, {})
        if not ds_spec:
            out[n] = list(pool)
            continue
        cls_to_idx = {name: i for i, name in enumerate(class_names_per_dataset[n])}
        # Validate class names
        for cls in ds_spec:
            if cls not in cls_to_idx:
                raise ValueError(
                    f"boost spec references unknown class {cls!r} for dataset {n!r}; "
                    f"available: {list(cls_to_idx.keys())}"
                )
        idx_factor = {cls_to_idx[c]: f for c, f in ds_spec.items()}
        new_pool: List[dict] = []
        for s in pool:
            new_pool.append(s)
            f = idx_factor.get(s["cls_label"], 1)
            if f > 1:
                # add (f-1) extra copies; rng.choices picks from the single-element list
                # but we want to keep this sample, so just append the dict ref directly
                for _ in range(f - 1):
                    new_pool.append(s)
        out[n] = new_pool
    return out


def apply_resampling(
    per_dataset_samples: Dict[str, List[dict]],
    *,
    strategy: str = "none",
    seed: int = 42,
    boost_spec: str = "",
    class_names_per_dataset: Dict[str, List[str]] | None = None,
) -> Dict[str, List[dict]]:
    """Apply a resampling strategy to per-dataset sample lists.

    Args:
      per_dataset_samples: {dataset_name: [sample_dict, ...]}
      strategy: "none" | "balance_datasets" | "balance_classes" | "boost"
      seed: RNG seed for reproducibility
      boost_spec: comma-separated "ds:class_name:factor" entries (only for "boost")
      class_names_per_dataset: required for "boost" (to map class names -> ids)

    Returns:
      Resampled per-dataset dict (same keys, possibly different list lengths).
    """
    rng = random.Random(seed)
    if strategy == "none":
        return {k: list(v) for k, v in per_dataset_samples.items()}
    if strategy == "balance_datasets":
        return _balance_datasets(per_dataset_samples, rng)
    if strategy == "balance_classes":
        return _balance_classes(per_dataset_samples, rng)
    if strategy == "boost":
        if class_names_per_dataset is None:
            raise ValueError("boost strategy requires class_names_per_dataset")
        return _boost_classes(per_dataset_samples, boost_spec, class_names_per_dataset, rng)
    raise ValueError(
        f"Unknown resample strategy: {strategy!r}. "
        f"Expected one of: none|balance_datasets|balance_classes|boost"
    )
