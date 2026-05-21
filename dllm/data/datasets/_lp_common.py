"""Dataset-agnostic LP split / negative-sampling / adj_train helpers.

Used by ``cora_lp.py``, ``arxiv_lp.py``, etc. The actual node-data + adjacency
load is delegated to each dataset's NC loader; this file only owns the
edge-level mechanics (split, neg sampling, leak prevention).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable

import torch


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


def load_lp_split(
    dataset_name: str,
    nc_loader: Callable[[dict, str, int], tuple],
    config: dict,
    split: str,
    seed: int = 42,
    neg_ratio: int = 1,
    val_frac: float = 0.05,
    test_frac: float = 0.10,
) -> dict:
    """Generic LP split loader.

    ``nc_loader(config, split, seed)`` must return
    ``(node_data, adj, class_names, split_ids)`` — the same signature used by
    each dataset's NC loader. The NC ``class_names`` are discarded; LP returns
    ``["no", "yes"]``.

    Returns the same dict shape as the original cora_lp.load().
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split={split!r}")

    cache_dir = _CACHE_ROOT / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"lp_split_seed{seed}_neg{neg_ratio}_v{int(val_frac*1000)}_t{int(test_frac*1000)}.pt"
    )

    node_data, adj_orig, _class_names_nc, _split_ids_nc = nc_loader(
        config, "train", seed
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


def _build_adj_from_edge_index(edge_index_tensor) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    ei = edge_index_tensor.cpu().numpy()
    for src, dst in zip(ei[0], ei[1]):
        u, v = int(src), int(dst)
        adj.setdefault(u, []).append(v)
    return adj


def load_lp_llaga_split(
    dataset_name: str,
    nc_loader: Callable[[dict, str, int], tuple],
    config: dict,
    split: str,
) -> dict:
    """Load LP edges from LLaGA's official train/test JSONL files.

    Uses ``edge_sampled_2_10_only_train.jsonl`` for training and
    ``edge_sampled_2_10_only_test.jsonl`` for evaluation.
    ``adj_train`` is built from ``processed_data_link_notest.pt`` (test edges
    already removed by LLaGA), so neighbor sampling during training never
    leaks test edges.

    Returns the same dict shape as ``load_lp_split``.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split={split!r}")

    # LLaGA only provides train + test; we map val -> train for convenience
    # (val is not used for HP selection in our LP SFT pipeline).
    jsonl_split = "train" if split in ("train", "val") else "test"
    data_dir = _CACHE_ROOT / dataset_name
    jsonl_path = data_dir / f"edge_sampled_2_10_only_{jsonl_split}.jsonl"
    notest_pt = data_dir / "processed_data_link_notest.pt"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"LLaGA LP JSONL not found: {jsonl_path}")
    if not notest_pt.exists():
        raise FileNotFoundError(f"LLaGA notest graph not found: {notest_pt}")

    pos_edges: list[tuple[int, int]] = []
    neg_edges: list[tuple[int, int]] = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            u, v = int(obj["id"][0]), int(obj["id"][1])
            label = obj["conversations"][1]["value"]
            if label == "yes":
                pos_edges.append((u, v))
            else:
                neg_edges.append((u, v))

    node_data, adj_orig, _cls, _ids = nc_loader(config, "train", seed=42)

    pt_data = torch.load(notest_pt, map_location="cpu", weights_only=False)
    adj_train = _build_adj_from_edge_index(pt_data.edge_index)

    n_nodes = max(node_data.keys()) + 1
    n_total_edges = len(_undirected_edge_set(adj_orig))

    return {
        "pos_edges": pos_edges,
        "neg_edges": neg_edges,
        "node_data": node_data,
        "adj_train": adj_train,
        "adj_orig": adj_orig,
        "class_names": ["no", "yes"],
        "n_nodes": n_nodes,
        "n_total_edges": n_total_edges,
        "split_counts": {
            "train_pos": len(pos_edges) if split == "train" else 0,
            "val_pos": 0,
            "test_pos": len(pos_edges) if split == "test" else 0,
            "train_neg": len(neg_edges) if split == "train" else 0,
            "val_neg": 0,
            "test_neg": len(neg_edges) if split == "test" else 0,
        },
    }
