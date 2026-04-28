"""ogbn-products TAPE subset loader (xxhe/tape-products: 54025 nodes, 42 classes used)."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch

from ._common import build_adjacency


def _stub_torch_sparse_for_unpickle():
    """Let torch.load unpickle SparseTensor without the native extension.

    We only need _row / _col from the storage; everything else can be a stub.
    """
    if "torch_sparse" in sys.modules:
        return

    ts_mod = types.ModuleType("torch_sparse")
    ts_mod.__path__ = []
    storage_mod = types.ModuleType("torch_sparse.storage")
    tensor_mod = types.ModuleType("torch_sparse.tensor")

    class _Stub:
        def __setstate__(self, s):
            if isinstance(s, dict):
                for k, v in s.items():
                    setattr(self, k, v)

    storage_mod.SparseStorage = _Stub
    tensor_mod.SparseTensor = _Stub
    ts_mod.SparseTensor = _Stub
    ts_mod.SparseStorage = _Stub
    sys.modules["torch_sparse"] = ts_mod
    sys.modules["torch_sparse.storage"] = storage_mod
    sys.modules["torch_sparse.tensor"] = tensor_mod


def load(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    _stub_torch_sparse_for_unpickle()

    tape_dir = Path(config["tape_dir"])
    class_names = config["class_names"]

    # 1. pt: graph + labels + splits
    data = torch.load(tape_dir / "ogbn-products_subset.pt", map_location="cpu", weights_only=False)
    y = data.y.squeeze(-1).tolist()
    train_mask = data.train_mask.tolist()
    val_mask = data.val_mask.tolist()
    test_mask = data.test_mask.tolist()

    storage = data.adj_t.storage
    src = np.array(storage._row.tolist())
    dst = np.array(storage._col.tolist())
    adj = build_adjacency(np.stack([src, dst], axis=0))

    # 2. CSV text in subset-local order (uid column matches pt n_asin[i])
    texts: list[dict] = []
    with open(tape_dir / "ogbn-products_subset.csv") as f:
        for row in csv.DictReader(f):
            texts.append(
                {
                    "title": (row.get("title") or "").strip(),
                    "abstract": (row.get("content") or "").strip(),
                }
            )
    assert len(texts) == len(y), f"CSV rows {len(texts)} != labels {len(y)}"

    # 3. node_data
    node_data: dict[int, dict] = {
        i: {"title": texts[i]["title"], "abstract": texts[i]["abstract"], "label": int(y[i])}
        for i in range(len(y))
    }

    # 4. Split
    mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
    split_ids = [i for i, m in enumerate(mask) if m]
    return node_data, adj, class_names, split_ids
