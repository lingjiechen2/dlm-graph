"""PubMed loader (local TAPE files: NODE.tab + pubmed.json + cites.tab)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
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

    tape_dir = Path(config["tape_dir"])
    class_names = config["class_names"]

    # 1. Parse node file: position -> (pmid, label)
    with open(tape_dir / "data" / "Pubmed-Diabetes.NODE.paper.tab") as f:
        lines = f.readlines()

    node_data: dict[int, dict] = {}
    all_ids: list[int] = []
    for i, line in enumerate(lines[2:]):
        parts = line.strip().split("\t")
        pmid = int(parts[0])
        label = int(parts[1].split("=")[1]) - 1  # 1-3 -> 0-2
        node_data[i] = {"pmid": pmid, "label": label, "title": "", "abstract": ""}
        all_ids.append(i)

    # 2. Load text from pubmed.json, matched by PMID
    with open(tape_dir / "pubmed.json") as f:
        pubmed_json = json.load(f)
    pmid_to_text = {
        int(e.get("PMID", 0)): {"title": e.get("TI", ""), "abstract": e.get("AB", "") or ""}
        for e in pubmed_json
    }
    for info in node_data.values():
        if info["pmid"] in pmid_to_text:
            info["title"] = pmid_to_text[info["pmid"]]["title"]
            info["abstract"] = pmid_to_text[info["pmid"]]["abstract"]

    # 3. Parse citations file -> undirected edge_index
    pmid_to_idx = {info["pmid"]: idx for idx, info in node_data.items()}
    edge_index = [[], []]
    with open(tape_dir / "data" / "Pubmed-Diabetes.DIRECTED.cites.tab") as f:
        cite_lines = f.readlines()
    for line in cite_lines[2:]:
        parts = line.strip().split("\t")
        src_pmid = int(parts[1].split(":")[1])
        dst_pmid = int(parts[3].split(":")[1])
        if src_pmid in pmid_to_idx and dst_pmid in pmid_to_idx:
            s, d = pmid_to_idx[src_pmid], pmid_to_idx[dst_pmid]
            edge_index[0].extend([s, d])
            edge_index[1].extend([d, s])
    adj = build_adjacency(np.array(edge_index))

    # 4. Stratified train/val/test split with fixed seed
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx in all_ids:
        by_label[node_data[idx]["label"]].append(idx)

    n_train, n_val, n_test = config["train_size"], config["val_size"], config["test_size"]
    n_classes = config["num_classes"]
    per_train = n_train // n_classes
    per_val = n_val // n_classes
    per_test = n_test // n_classes

    train_ids, val_ids, test_ids = [], [], []
    for lbl in range(n_classes):
        indices = by_label[lbl][:]
        rng.shuffle(indices)
        train_ids.extend(indices[:per_train])
        val_ids.extend(indices[per_train : per_train + per_val])
        test_ids.extend(indices[per_train + per_val : per_train + per_val + per_test])

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]
    return node_data, adj, class_names, split_ids
