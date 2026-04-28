"""ogbn-arxiv loader (OGB raw files + titleabs.tsv, with optional LLaGA cache)."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from ._common import build_adjacency, load_llaga_processed_data


def load(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    llaga = load_llaga_processed_data(config, split)
    if llaga is not None:
        return llaga

    ogb_root = Path(config["ogb_root"])
    class_names = config["class_names"]

    # 1. Node index -> paper ID
    nodeidx_to_paperid: dict[int, int] = {}
    with gzip.open(ogb_root / "mapping" / "nodeidx2paperid.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            nodeidx_to_paperid[int(row[0])] = int(row[1])

    # 2. Paper ID -> (title, abstract)
    paperid_to_text: dict[int, dict] = {}
    with open(ogb_root / "raw" / "titleabs.tsv") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                paperid_to_text[int(parts[0])] = {"title": parts[1], "abstract": parts[2]}
            elif len(parts) == 2:
                paperid_to_text[int(parts[0])] = {"title": parts[1], "abstract": ""}

    # 3. Node labels
    node_labels: dict[int, int] = {}
    with gzip.open(ogb_root / "raw" / "node-label.csv.gz", "rt") as f:
        for idx, line in enumerate(f):
            node_labels[idx] = int(line.strip())

    # 4. Edges (undirected)
    src_list, dst_list = [], []
    with gzip.open(ogb_root / "raw" / "edge.csv.gz", "rt") as f:
        for line in f:
            parts = line.strip().split(",")
            s, d = int(parts[0]), int(parts[1])
            src_list.extend([s, d])
            dst_list.extend([d, s])
    adj = build_adjacency(np.array([src_list, dst_list]))

    # 5. Split
    split_name = {"train": "train", "val": "valid", "test": "test"}[split]
    split_ids: list[int] = []
    with gzip.open(ogb_root / "split" / "time" / f"{split_name}.csv.gz", "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("node"):
                split_ids.append(int(line))

    # 6. node_data
    node_data: dict[int, dict] = {}
    for node_idx, paper_id in nodeidx_to_paperid.items():
        text = paperid_to_text.get(paper_id, {"title": "", "abstract": ""})
        node_data[node_idx] = {
            "title": text["title"],
            "abstract": text["abstract"],
            "label": node_labels.get(node_idx, 0),
        }

    return node_data, adj, class_names, split_ids
