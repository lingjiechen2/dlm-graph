"""Cora link-prediction loader.

Builds a fixed train/val/test edge split from the cora adjacency, samples
uniform negatives at the requested ratio, and returns the train-only adjacency
(val+test positive edges removed) so neighbour sampling does not leak the
target edge into the prompt.

The split is cached deterministically per ``seed`` and ``neg_ratio`` so
re-runs see identical edges.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch

from . import cora as cora_nc

_CACHE_ROOT = Path(
    os.environ.get(
        "LLAGA_DATA_ROOT",
        "/home/lingjie7/auto-research/projects/dlm-graph/.datasets/llaga",
    )
)


def _undirected_edge_set(adj: dict[int, list[int]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for u, vs in adj.items():
        for v in vs:
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    return edges


def _split_edges(
    edges: list[tuple[int, int]],
    seed: int,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
) -> tuple[list, list, list]:
    rng = random.Random(seed)
    edges = list(edges)
    rng.shuffle(edges)
    n = len(edges)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = edges[:n_val]
    test = edges[n_val : n_val + n_test]
    train = edges[n_val + n_test :]
    return train, val, test


def _sample_negatives(
    n_nodes: int,
    pos_edge_set: set[tuple[int, int]],
    n_negatives: int,
    seed: int,
    avoid_edges: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Uniform-random non-edge sampling. ``avoid_edges`` adds extra pairs to
    skip (e.g. negatives sampled for other splits) so train/val/test negatives
    are mutually disjoint."""
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set(avoid_edges or set())
    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n_negatives * 50
    while len(out) < n_negatives and attempts < max_attempts:
        attempts += 1
        u = rng.randrange(n_nodes)
        v = rng.randrange(n_nodes)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key in pos_edge_set or key in seen:
            continue
        seen.add(key)
        out.append(key)
    if len(out) < n_negatives:
        raise RuntimeError(
            f"Could not sample {n_negatives} negatives in {max_attempts} attempts; "
            f"got {len(out)}. Graph may be too dense."
        )
    return out


def _adj_remove_edges(
    adj: dict[int, list[int]],
    edges_to_remove: set[tuple[int, int]],
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for u, vs in adj.items():
        kept = []
        for v in vs:
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges_to_remove:
                continue
            kept.append(v)
        if kept:
            out[u] = kept
    return out


def load(
    config: dict,
    split: str,
    seed: int = 42,
    neg_ratio: int = 1,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
) -> dict:
    """Load the cora LP split and return:

    {
      "pos_edges":  list[(u, v)] for this split,
      "neg_edges":  list[(u, v)] for this split (1:neg_ratio per positive),
      "node_data":  dict[node_id, {title, abstract, label}],
      "adj_train":  adj with all val+test positive edges removed,
      "class_names": ["no", "yes"],
    }

    Note: the original cora ``label`` (paper category) is kept inside
    ``node_data`` for free, but is not used for LP training.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split={split!r}")

    cache_dir = _CACHE_ROOT / "cora"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"lp_split_seed{seed}_neg{neg_ratio}_v{int(val_frac*1000)}_t{int(test_frac*1000)}.pt"
    )

    node_data, adj_orig, _class_names_nc, _split_ids_nc = cora_nc.load(
        config, split="train", seed=seed
    )

    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
    else:
        all_edges = sorted(_undirected_edge_set(adj_orig))
        train_edges, val_edges, test_edges = _split_edges(
            all_edges, seed=seed, val_frac=val_frac, test_frac=test_frac
        )
        pos_set = set(all_edges)
        n_nodes = max(node_data.keys()) + 1
        neg_train = _sample_negatives(
            n_nodes, pos_set, len(train_edges) * neg_ratio, seed=seed * 31 + 1
        )
        neg_val = _sample_negatives(
            n_nodes,
            pos_set,
            len(val_edges) * neg_ratio,
            seed=seed * 31 + 2,
            avoid_edges=set(neg_train),
        )
        neg_test = _sample_negatives(
            n_nodes,
            pos_set,
            len(test_edges) * neg_ratio,
            seed=seed * 31 + 3,
            avoid_edges=set(neg_train) | set(neg_val),
        )
        cache = {
            "train_pos": train_edges,
            "val_pos": val_edges,
            "test_pos": test_edges,
            "train_neg": neg_train,
            "val_neg": neg_val,
            "test_neg": neg_test,
            "n_nodes": n_nodes,
            "n_total_edges": len(all_edges),
        }
        torch.save(cache, cache_path)

    val_test_pos_set: set[tuple[int, int]] = set(cache["val_pos"]) | set(cache["test_pos"])
    adj_train = _adj_remove_edges(adj_orig, val_test_pos_set)

    pos_edges = list(cache[f"{split}_pos"])
    neg_edges = list(cache[f"{split}_neg"])

    return {
        "pos_edges": pos_edges,
        "neg_edges": neg_edges,
        "node_data": node_data,
        "adj_train": adj_train,
        "adj_orig": adj_orig,
        "class_names": ["no", "yes"],
        "n_nodes": cache["n_nodes"],
        "n_total_edges": cache["n_total_edges"],
        "split_counts": {
            "train_pos": len(cache["train_pos"]),
            "val_pos": len(cache["val_pos"]),
            "test_pos": len(cache["test_pos"]),
            "train_neg": len(cache["train_neg"]),
            "val_neg": len(cache["val_neg"]),
            "test_neg": len(cache["test_neg"]),
        },
    }
