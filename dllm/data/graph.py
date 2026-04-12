"""
TAG (Text-Attributed Graph) dataset loader for TM-DLM.

Loads graph datasets (Cora, ogbn-arxiv) with raw text from TAPE benchmark
and graph structure from PyG, then converts them into the S_v sequence
format expected by the TM-DLM pipeline.

Each returned sample contains:
    input_ids       list[int]       Tokenized S_v sequence
    labels          list[int]       -100 everywhere except class-name token positions
    node_spans      list[[int,int]] Token span per node (target first, then neighbors)
    node_hops       list[int]       Hop distance per node
    cls_label       int             Integer class label
    label_token_pos int             Index of first class-name token in input_ids
"""

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset, Dataset

# Default HuggingFace cache root
HF_CACHE_ROOT = Path(
    os.environ.get("HF_DATASETS_CACHE", Path.home() / "datasets" / "huggingface")
)

DATASET_CONFIGS = {
    "cora": {
        "hf_path": "xxhe/tape-cora",
        "cache_dir": HF_CACHE_ROOT / "xxhe___tape-cora",
        "pyg_root": HF_CACHE_ROOT / "pyg___cora",
        "pyg_name": "Cora",
        "num_classes": 7,
    },
    "pubmed": {
        "tape_dir": HF_CACHE_ROOT / "tape___pubmed" / "PubMed_orig",
        "num_classes": 3,
        "class_names": [
            "Diabetes Mellitus, Experimental",
            "Diabetes Mellitus Type 1",
            "Diabetes Mellitus Type 2",
        ],
        # Standard Planetoid split sizes
        "train_size": 60,
        "val_size": 500,
        "test_size": 1000,
    },
}

# Neighbor sampling config (aligned with LLaGA)
MAX_NEIGHBORS_PER_HOP = 10
MAX_HOPS = 2


def _build_adjacency(edge_index: np.ndarray) -> dict[int, list[int]]:
    """Build adjacency list from edge_index [2, E] numpy array."""
    adj = defaultdict(list)
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        adj[src].append(dst)
    return dict(adj)


def _sample_khop_neighbors(
    adj: dict[int, list[int]],
    node_id: int,
    max_neighbors_per_hop: int,
    max_hops: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """
    Sample k-hop neighbors for a node.

    Returns:
        neighbor_ids: list of neighbor node IDs
        neighbor_hops: list of hop distances (1 or 2)
    """
    neighbor_ids = []
    neighbor_hops = []
    seen = {node_id}

    if max_hops < 1:
        return neighbor_ids, neighbor_hops

    # 1-hop
    nb_1hop = list(set(adj.get(node_id, [])))
    if len(nb_1hop) > max_neighbors_per_hop:
        nb_1hop = rng.sample(nb_1hop, max_neighbors_per_hop)
    for nb in nb_1hop:
        seen.add(nb)
    neighbor_ids.extend(nb_1hop)
    neighbor_hops.extend([1] * len(nb_1hop))

    # 2-hop
    if max_hops >= 2:
        candidates_2hop = []
        for nb in nb_1hop:
            for nb2 in adj.get(nb, []):
                if nb2 not in seen:
                    candidates_2hop.append(nb2)
                    seen.add(nb2)
        if len(candidates_2hop) > max_neighbors_per_hop:
            candidates_2hop = rng.sample(candidates_2hop, max_neighbors_per_hop)
        neighbor_ids.extend(candidates_2hop)
        neighbor_hops.extend([2] * len(candidates_2hop))

    return neighbor_ids, neighbor_hops


def build_node_sample(
    target_node_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    class_names: list[str],
    tokenizer,
    max_seq_len: int = 2048,
    mask_target_text: bool = False,
) -> dict:
    """
    Build a single TM-DLM training sample for one target node.

    Uses a multiple-choice format with single-digit answer to avoid
    multi-token class name issues:

        Paper: <target_text>
        Options: 0) Case Based 1) Genetic Algorithms ... 6) Theory
        Answer: <digit>
        Neighbor 1: <nb1_text>
        ...

    The target span (including options and answer) is node_spans[0] with hop=0.
    Each neighbor is a separate span with its hop distance.

    Labels are -100 everywhere EXCEPT at the answer digit position.
    """

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # --- Build options string ---
    options_str = " ".join(f"{i}) {name}" for i, name in enumerate(class_names))
    answer_str = str(cls_label)

    # --- Tokenize target section ---
    target_prefix = _tok("Paper: ")
    target_body = _tok(target_node_text)
    options_prefix = _tok(f"\nOptions: {options_str}\nAnswer: ")
    answer_tokens = _tok(answer_str)  # single digit → typically 1 token

    # Reserve space: target gets up to half of max_seq_len
    overhead = len(target_prefix) + len(options_prefix) + len(answer_tokens)
    target_body_budget = max(max_seq_len // 2 - overhead, 50)
    target_body = target_body[:target_body_budget]

    # Build target + options + answer section
    input_ids = target_prefix + target_body + options_prefix + answer_tokens
    label_token_pos = len(target_prefix) + len(target_body) + len(options_prefix)
    target_span_end = len(input_ids)

    node_spans = [[0, target_span_end]]
    node_hops_out = [0]

    # --- Tokenize neighbor sections ---
    num_neighbors = len(neighbor_texts)
    remaining_budget = max_seq_len - len(input_ids)
    per_nb_budget = (
        max(remaining_budget // max(num_neighbors, 1), 20) if num_neighbors > 0 else 0
    )

    for i, (nb_text, hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        if len(input_ids) >= max_seq_len:
            break

        nb_prefix = _tok(f"\nNeighbor {i + 1}: ")
        nb_body = _tok(nb_text)
        nb_ids = (nb_prefix + nb_body)[:per_nb_budget]

        start = len(input_ids)
        input_ids.extend(nb_ids)
        node_spans.append([start, len(input_ids)])
        node_hops_out.append(hop)

    # Enforce max_seq_len
    input_ids = input_ids[:max_seq_len]

    # --- Labels: -100 everywhere except masked positions ---
    labels = [-100] * len(input_ids)
    if mask_target_text:
        # Mask all target body + answer tokens (dense training signal)
        target_body_start = len(target_prefix)
        target_body_end = len(target_prefix) + len(target_body)
        for pos in range(target_body_start, min(target_body_end, len(labels))):
            labels[pos] = input_ids[pos]
    # Always include answer digit in labels
    for j in range(len(answer_tokens)):
        pos = label_token_pos + j
        if pos < len(labels):
            labels[pos] = input_ids[pos]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
    }


def _load_pubmed_from_tape(config: dict, split: str, seed: int = 42):
    """
    Load PubMed data from local TAPE files (tab + JSON + citations).

    Returns:
        node_data: dict[int, dict] with title, abstract, label for ALL nodes
        adj: adjacency list
        class_names: list of class name strings
        split_ids: list of node IDs for the requested split
    """
    import json as _json

    tape_dir = Path(config["tape_dir"])
    class_names = config["class_names"]

    # 1. Parse node file: position → (pmid, label)
    node_file = tape_dir / "data" / "Pubmed-Diabetes.NODE.paper.tab"
    with open(node_file) as f:
        lines = f.readlines()

    node_data: dict[int, dict] = {}
    all_ids = []
    for i, line in enumerate(lines[2:]):
        parts = line.strip().split("\t")
        pmid = int(parts[0])
        label = int(parts[1].split("=")[1]) - 1  # 1-3 → 0-2
        node_data[i] = {"pmid": pmid, "label": label, "title": "", "abstract": ""}
        all_ids.append(i)

    # 2. Load text from pubmed.json, matched by PMID
    json_file = tape_dir / "pubmed.json"
    with open(json_file) as f:
        pubmed_json = _json.load(f)

    pmid_to_text = {}
    for entry in pubmed_json:
        pmid = int(entry.get("PMID", 0))
        pmid_to_text[pmid] = {
            "title": entry.get("TI", ""),
            "abstract": entry.get("AB", "") or "",
        }

    for idx, info in node_data.items():
        if info["pmid"] in pmid_to_text:
            info["title"] = pmid_to_text[info["pmid"]]["title"]
            info["abstract"] = pmid_to_text[info["pmid"]]["abstract"]

    # 3. Parse citations file for edges
    cite_file = tape_dir / "data" / "Pubmed-Diabetes.DIRECTED.cites.tab"
    pmid_to_idx = {info["pmid"]: idx for idx, info in node_data.items()}

    with open(cite_file) as f:
        cite_lines = f.readlines()

    edge_index = [[], []]
    for line in cite_lines[2:]:
        parts = line.strip().split("\t")
        src_pmid = int(parts[1].split(":")[1])
        dst_pmid = int(parts[3].split(":")[1])
        if src_pmid in pmid_to_idx and dst_pmid in pmid_to_idx:
            s, d = pmid_to_idx[src_pmid], pmid_to_idx[dst_pmid]
            edge_index[0].extend([s, d])
            edge_index[1].extend([d, s])

    adj = _build_adjacency(np.array(edge_index))

    # 4. Create stratified train/val/test split with fixed seed
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx in all_ids:
        by_label[node_data[idx]["label"]].append(idx)

    train_ids, val_ids, test_ids = [], [], []
    n_train = config["train_size"]
    n_val = config["val_size"]
    n_test = config["test_size"]
    n_classes = config["num_classes"]

    for lbl in range(n_classes):
        indices = by_label[lbl][:]
        rng.shuffle(indices)
        per_class_train = n_train // n_classes
        per_class_val = n_val // n_classes
        per_class_test = n_test // n_classes
        train_ids.extend(indices[:per_class_train])
        val_ids.extend(indices[per_class_train : per_class_train + per_class_val])
        test_ids.extend(
            indices[
                per_class_train
                + per_class_val : per_class_train
                + per_class_val
                + per_class_test
            ]
        )

    split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
    return node_data, adj, class_names, split_map[split]


def load_tag_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    max_seq_len: int = 2048,
    max_neighbors_per_hop: int = MAX_NEIGHBORS_PER_HOP,
    max_hops: int = MAX_HOPS,
    seed: int = 42,
    mask_target_text: bool = False,
) -> Dataset:
    """
    Load a TAG dataset and return a HuggingFace Dataset of TM-DLM samples.

    Supports: "cora" (from HuggingFace + PyG), "pubmed" (from local TAPE files).
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == "pubmed":
        return _load_pubmed_tag_dataset(
            config,
            tokenizer,
            split,
            max_seq_len,
            max_neighbors_per_hop,
            max_hops,
            seed,
            mask_target_text,
        )

    # --- Cora path (HuggingFace + PyG) ---
    split_map = {"train": "train", "val": "validation", "test": "test"}
    hf_split = split_map[split]

    # --- 1. Load TAPE text data ---
    tape_ds = load_dataset(
        config["hf_path"],
        cache_dir=str(config["cache_dir"]),
    )

    # Build full text lookup: node_id -> info
    node_data: dict[int, dict] = {}
    for s in tape_ds:
        for sample in tape_ds[s]:
            node_data[sample["id"]] = {
                "title": sample["T"],
                "abstract": sample["A"],
                "label": sample["label"],
                "class_name": sample["class"],
            }

    # --- 2. Load graph structure from PyG Planetoid ---
    from torch_geometric.datasets import Planetoid

    pyg_root = str(config["pyg_root"])
    pyg_ds = Planetoid(root=pyg_root, name=config["pyg_name"])
    edge_index = pyg_ds[0].edge_index.numpy()

    # --- 3. Build adjacency ---
    adj = _build_adjacency(edge_index)

    # --- 4. Build class-name list (sorted by class index) ---
    label_to_class = {}
    for info in node_data.values():
        label_to_class[info["label"]] = info["class_name"]
    class_names = [label_to_class[i] for i in range(len(label_to_class))]

    # --- 5. Process each node in target split ---
    rng = random.Random(seed)
    samples = []

    for sample in tape_ds[hf_split]:
        node_id = sample["id"]
        info = node_data[node_id]

        nb_ids, nb_hops = _sample_khop_neighbors(
            adj, node_id, max_neighbors_per_hop, max_hops, rng
        )

        neighbor_texts = []
        neighbor_hops = []
        for nb_id, hop in zip(nb_ids, nb_hops):
            if nb_id in node_data:
                nd = node_data[nb_id]
                neighbor_texts.append(f"{nd['title']}. {nd['abstract']}")
                neighbor_hops.append(hop)

        target_text = f"{info['title']}. {info['abstract']}"

        result = build_node_sample(
            target_node_text=target_text,
            neighbor_texts=neighbor_texts,
            neighbor_hops=neighbor_hops,
            cls_label=info["label"],
            class_names=class_names,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_target_text=mask_target_text,
        )
        samples.append(result)

    dataset = Dataset.from_list(samples)
    dataset.info.description = (
        f"TM-DLM {dataset_name} ({split}), classes: {class_names}"
    )
    return dataset


def _load_pubmed_tag_dataset(
    config,
    tokenizer,
    split,
    max_seq_len,
    max_neighbors_per_hop,
    max_hops,
    seed,
    mask_target_text,
) -> Dataset:
    """Load PubMed TAG dataset from local TAPE files."""
    node_data, adj, class_names, split_ids = _load_pubmed_from_tape(config, split, seed)

    rng = random.Random(seed)
    samples = []

    for node_id in split_ids:
        info = node_data[node_id]

        nb_ids, nb_hops = _sample_khop_neighbors(
            adj, node_id, max_neighbors_per_hop, max_hops, rng
        )

        neighbor_texts = []
        neighbor_hops = []
        for nb_id, hop in zip(nb_ids, nb_hops):
            if nb_id in node_data:
                nd = node_data[nb_id]
                neighbor_texts.append(f"{nd['title']}. {nd['abstract']}")
                neighbor_hops.append(hop)

        target_text = f"{info['title']}. {info['abstract']}"

        result = build_node_sample(
            target_node_text=target_text,
            neighbor_texts=neighbor_texts,
            neighbor_hops=neighbor_hops,
            cls_label=info["label"],
            class_names=class_names,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_target_text=mask_target_text,
        )
        samples.append(result)

    dataset = Dataset.from_list(samples)
    dataset.info.description = f"TM-DLM pubmed ({split}), classes: {class_names}"
    return dataset


def get_class_token_ids(dataset_name: str, tokenizer) -> tuple[list[str], list[int]]:
    """
    Get the answer digit token IDs for multiple-choice classification.

    With the MC format, answers are "0", "1", ..., "K-1" (single digits).

    Returns:
        class_names: list of class name strings, ordered by class index
        answer_token_ids: list of token IDs for "0", "1", ..., "K-1"
    """
    config = DATASET_CONFIGS[dataset_name]

    if dataset_name == "pubmed":
        class_names = config["class_names"]
    else:
        tape_ds = load_dataset(
            config["hf_path"],
            cache_dir=str(config["cache_dir"]),
        )
        label_to_class = {}
        for s in tape_ds:
            for sample in tape_ds[s]:
                label_to_class[sample["label"]] = sample["class"]
        class_names = [label_to_class[i] for i in range(len(label_to_class))]

    # Answer tokens are digits: "0", "1", ..., "K-1"
    answer_token_ids = []
    for i in range(len(class_names)):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        answer_token_ids.append(tokens[0])

    return class_names, answer_token_ids
