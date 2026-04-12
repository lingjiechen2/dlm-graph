"""
TAG (Text-Attributed Graph) dataset loader for TM-DLM.

Loads graph datasets (Cora, ogbn-arxiv, ogbn-products) and converts
them into the S_v sequence format expected by the TM-DLM pipeline.

Dataset storage convention: ~/datasets/huggingface/{org}___{name}
  - ogbn-arxiv  -> snap-stanford___ogbn-arxiv
  - Cora        -> pyg-data___cora
  - ogbn-products -> snap-stanford___ogbn-products

Each returned sample contains:
    input_ids       list[int]       Tokenized S_v sequence
    labels          list[int]       Same, with -100 at neighbor token positions
    node_spans      list[(int,int)] Token span per node (target first, then neighbors)
    node_hops       list[int]       Hop distance per node
    cls_label       int             Integer class label
    label_token_pos int             Index of [LABEL] token in input_ids
"""

import os
from pathlib import Path
from typing import Callable

import torch
from datasets import DatasetDict, Dataset

# Default HuggingFace cache root
HF_CACHE_ROOT = Path(os.environ.get("HF_DATASETS_CACHE", Path.home() / "datasets" / "huggingface"))

DATASET_CONFIGS = {
    "cora": {
        "hf_path": "pyg-data/cora",
        "cache_dir": HF_CACHE_ROOT / "pyg-data___cora",
        "num_classes": 7,
        "text_fields": ["title", "abstract"],
    },
    "pubmed": {
        "hf_path": "pyg-data/pubmed",
        "cache_dir": HF_CACHE_ROOT / "pyg-data___pubmed",
        "num_classes": 3,
        "text_fields": ["title", "abstract"],
    },
    "ogbn-arxiv": {
        "hf_path": "snap-stanford/ogbn-arxiv",
        "cache_dir": HF_CACHE_ROOT / "snap-stanford___ogbn-arxiv",
        "num_classes": 40,
        "text_fields": ["title", "abstract"],
    },
    "ogbn-products": {
        "hf_path": "snap-stanford/ogbn-products",
        "cache_dir": HF_CACHE_ROOT / "snap-stanford___ogbn-products",
        "num_classes": 47,
        "text_fields": ["title", "content"],
    },
}

# Neighbor sampling config (aligned with LLaGA)
MAX_NEIGHBORS_PER_HOP = 10
MAX_HOPS = 2


def load_tag_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    max_seq_len: int = 2048,
    max_neighbors_per_hop: int = MAX_NEIGHBORS_PER_HOP,
    max_hops: int = MAX_HOPS,
    seed: int = 42,
) -> Dataset:
    """
    Load a TAG dataset and return a HuggingFace Dataset of TM-DLM samples.

    Args:
        dataset_name:  One of "cora", "pubmed", "ogbn-arxiv", "ogbn-products"
        tokenizer:     HuggingFace tokenizer (must have mask_token_id)
        split:         "train", "val", or "test"
        max_seq_len:   Maximum sequence length for S_v
        max_neighbors_per_hop: Number of neighbors sampled per hop
        max_hops:      Number of hops (1 or 2)
        seed:          Random seed for neighbor sampling

    Returns:
        HuggingFace Dataset with fields: input_ids, labels, node_spans,
        node_hops, cls_label, label_token_pos
    """
    raise NotImplementedError(
        "load_tag_dataset is not yet implemented. "
        "Implement graph loading with PyG or OGB and call build_node_sample() "
        "for each node to construct the S_v sequence."
    )


def build_node_sample(
    target_node_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    class_name: str,
    tokenizer,
    max_seq_len: int = 2048,
) -> dict:
    """
    Build a single TM-DLM training sample for one target node.

    Constructs:
        S_v = [TARGET] <target_text> [SEP] [NODE_1] <nb1_text> ... [SEP] [NODE_m] <nbm_text> [LABEL]

    Args:
        target_node_text:   Raw text of the target node
        neighbor_texts:     Texts of sampled neighbors (in order)
        neighbor_hops:      Hop distance for each neighbor
        cls_label:          Integer class index
        class_name:         String class name (e.g. "machine learning")
        tokenizer:          HuggingFace tokenizer

    Returns:
        dict with input_ids, labels, node_spans, node_hops, cls_label, label_token_pos
    """
    # Special tokens (added if not present in vocab)
    TARGET_TOKEN = "[TARGET]"
    SEP_TOKEN = "[SEP]"
    LABEL_TOKEN = "[LABEL]"

    input_ids = []
    node_spans = []
    node_hops_out = []

    def _tokenize(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # --- Target node ---
    target_prefix = tokenizer.convert_tokens_to_ids(TARGET_TOKEN)
    target_ids = [target_prefix] + _tokenize(target_node_text)
    start = len(input_ids)
    input_ids.extend(target_ids)
    node_spans.append((start, len(input_ids)))
    node_hops_out.append(0)

    # --- Neighbor nodes ---
    for nb_idx, (nb_text, nb_hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        node_prefix_id = tokenizer.convert_tokens_to_ids(f"[NODE_{nb_idx + 1}]")
        nb_ids = [sep_id, node_prefix_id] + _tokenize(nb_text)

        start = len(input_ids)
        # Truncate if adding this neighbor exceeds max_seq_len (leave room for label)
        remaining = max_seq_len - len(input_ids) - 2  # -2 for label token + SEP
        if remaining <= 0:
            break
        nb_ids = nb_ids[:remaining]
        input_ids.extend(nb_ids)
        node_spans.append((start, len(input_ids)))
        node_hops_out.append(nb_hop)

    # --- Label token ---
    label_id = tokenizer.convert_tokens_to_ids(LABEL_TOKEN)
    label_pos = len(input_ids)
    input_ids.append(label_id)

    # --- Labels (for training loss) ---
    # -100 at neighbor token positions; real token ids at target + label positions
    labels = []
    target_start, target_end = node_spans[0]
    for pos, tok in enumerate(input_ids):
        in_target = target_start <= pos < target_end
        is_label = pos == label_pos
        if in_target or is_label:
            labels.append(tok)
        else:
            labels.append(-100)

    # Override label position with the tokenized class name token
    class_token_ids = _tokenize(class_name)
    if class_token_ids:
        labels[label_pos] = class_token_ids[0]
        input_ids[label_pos] = class_token_ids[0]  # replace [LABEL] with actual class token

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_pos,
    }
