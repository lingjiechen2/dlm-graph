"""Shared helpers for per-dataset loaders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def build_adjacency(edge_index: np.ndarray) -> dict[int, list[int]]:
    """Build adjacency list from edge_index [2, E]."""
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        adj[int(edge_index[0, i])].append(int(edge_index[1, i]))
    return dict(adj)


_PUBMED_LLAGA_NAME_MAP = {
    "Diabetes Mellitus Experimental": "Diabetes Mellitus, Experimental",
    "Diabetes Mellitus Type1": "Diabetes Mellitus Type 1",
    "Diabetes Mellitus Type2": "Diabetes Mellitus Type 2",
}


def normalize_llaga_class_name(name: str) -> str:
    name = name.replace("_", " ").strip()
    return _PUBMED_LLAGA_NAME_MAP.get(name, name)


def load_llaga_processed_data(
    config: dict,
    split: str,
) -> tuple[dict, dict, list[str], list[int]] | None:
    """Load LLaGA's processed_data.pt if it exists for this dataset.

    Returns (node_data, adj, class_names, split_ids) or None if no LLaGA cache.
    """
    llaga_dir = config.get("llaga_dir")
    if llaga_dir is None:
        return None
    data_path = Path(llaga_dir) / "processed_data.pt"
    if not data_path.exists():
        return None

    data = torch.load(data_path, map_location="cpu")
    edge_index = data["edge_index"]
    if hasattr(edge_index, "cpu"):
        edge_index = edge_index.cpu().numpy()
    adj = build_adjacency(edge_index)

    class_names = [normalize_llaga_class_name(x) for x in list(data["label_texts"])]

    n = int(data["num_nodes"])
    titles = list(data["title"]) if "title" in data else [""] * n
    abstracts = list(data["abs"]) if "abs" in data else [""] * n
    labels = data["y"]
    if hasattr(labels, "cpu"):
        labels = labels.cpu().tolist()
    else:
        labels = list(labels)

    node_data: dict[int, dict] = {
        idx: {
            "title": titles[idx],
            "abstract": abstracts[idx],
            "label": int(labels[idx]),
        }
        for idx in range(n)
    }

    split_key = {"train": "train_id", "val": "val_id", "test": "test_id"}[split]
    if split_key in data:
        split_ids = data[split_key]
        if hasattr(split_ids, "cpu"):
            split_ids = split_ids.cpu().tolist()
        else:
            split_ids = list(split_ids)
    else:
        mask_key = {"train": "train_mask", "val": "val_mask", "test": "test_mask"}[split]
        split_ids = data[mask_key].nonzero(as_tuple=False).view(-1).cpu().tolist()
    split_ids = [int(x) for x in split_ids]
    return node_data, adj, class_names, split_ids
