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

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from datasets import load_dataset, Dataset

from .datasets import LOADERS as _DATA_LOADERS, load_llaga_processed_data as _load_llaga_processed_data

logger = logging.getLogger(__name__)

# Default HuggingFace cache root
HF_CACHE_ROOT = Path(
    os.environ.get("HF_DATASETS_CACHE", Path.home() / "datasets" / "huggingface")
)
# Persistent on-disk cache for built TAG datasets. Keyed by all args that
# affect sample content (dataset, tokenizer, prompt format, sampling seed,
# etc.). DDP rank 0 builds, ranks 1+ load from disk in seconds.
TAG_CACHE_ROOT = Path(
    os.environ.get("TMDLM_TAG_CACHE_ROOT", HF_CACHE_ROOT / "tmdlm_tag_cache")
)
_TAG_CACHE_VERSION = 2  # bumped for arxiv cs.CL fix + cora double-prefix strip
LLAGA_DATA_ROOT = Path(
    os.environ.get(
        "LLAGA_DATA_ROOT",
        "/home/lingjie7/auto-research/projects/dlm-graph/.datasets/llaga",
    )
)

DATASET_CONFIGS = {
    "cora": {
        "hf_path": "xxhe/tape-cora",
        "cache_dir": HF_CACHE_ROOT / "xxhe___tape-cora",
        "pyg_root": HF_CACHE_ROOT / "pyg___cora",
        "llaga_dir": LLAGA_DATA_ROOT / "cora",
        "pyg_name": "Cora",
        "num_classes": 7,
    },
    "pubmed": {
        "tape_dir": HF_CACHE_ROOT / "tape___pubmed" / "PubMed_orig",
        "llaga_dir": LLAGA_DATA_ROOT / "pubmed",
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
    "ogbn-products": {
        "tape_dir": HF_CACHE_ROOT / "xxhe___tape-products",
        "num_classes": 47,
        # From labelidx2productcategory.csv.gz (OGB). Empty/sentinel rows
        # replaced with placeholders so tokenization stays unambiguous.
        "class_names": [
            "Home & Kitchen",  # 0
            "Health & Personal Care",  # 1
            "Beauty",  # 2
            "Sports & Outdoors",  # 3
            "Books",  # 4
            "Patio, Lawn & Garden",  # 5
            "Toys & Games",  # 6
            "CDs & Vinyl",  # 7
            "Cell Phones & Accessories",  # 8
            "Grocery & Gourmet Food",  # 9
            "Arts, Crafts & Sewing",  # 10
            "Clothing, Shoes & Jewelry",  # 11
            "Electronics",  # 12
            "Movies & TV",  # 13
            "Software",  # 14
            "Video Games",  # 15
            "Automotive",  # 16
            "Pet Supplies",  # 17
            "Office Products",  # 18
            "Industrial & Scientific",  # 19
            "Musical Instruments",  # 20
            "Tools & Home Improvement",  # 21
            "Magazine Subscriptions",  # 22
            "Baby Products",  # 23
            "Category 24",  # 24 (empty in OGB mapping)
            "Appliances",  # 25
            "Kitchen & Dining",  # 26
            "Collectibles & Fine Art",  # 27
            "All Beauty",  # 28
            "Luxury Beauty",  # 29
            "Amazon Fashion",  # 30
            "Computers",  # 31
            "All Electronics",  # 32
            "Purchase Circles",  # 33
            "MP3 Players & Accessories",  # 34
            "Gift Cards",  # 35
            "Office & School Supplies",  # 36
            "Home Improvement",  # 37
            "Camera & Photo",  # 38
            "GPS & Navigation",  # 39
            "Digital Music",  # 40
            "Car Electronics",  # 41
            "Baby",  # 42
            "Kindle Store",  # 43
            "Buy a Kindle",  # 44
            "Furniture & Decor",  # 45
            "Category 46",  # 46 ("#508510" sentinel in OGB mapping)
        ],
    },
    "ogbn-arxiv": {
        "ogb_root": HF_CACHE_ROOT / "ogb___ogbn-arxiv" / "ogbn_arxiv",
        "llaga_dir": LLAGA_DATA_ROOT / "ogbn-arxiv",
        "num_classes": 40,
        # Full category names (mapped from labelidx2arxivcategory.csv.gz order)
        "class_names": [
            "Numerical Analysis",  # 0: cs.NA
            "Multimedia",  # 1: cs.MM
            "Logic in Computer Science",  # 2: cs.LO
            "Computers and Society",  # 3: cs.CY
            "Cryptography and Security",  # 4: cs.CR
            "Distributed, Parallel, and Cluster Computing",  # 5: cs.DC
            "Human-Computer Interaction",  # 6: cs.HC
            "Computational Engineering, Finance, and Science",  # 7: cs.CE
            "Networking and Internet Architecture",  # 8: cs.NI
            "Computational Complexity",  # 9: cs.CC
            "Artificial Intelligence",  # 10: cs.AI
            "Multiagent Systems",  # 11: cs.MA
            "General Literature",  # 12: cs.GL
            "Neural and Evolutionary Computing",  # 13: cs.NE
            "Symbolic Computation",  # 14: cs.SC
            "Hardware Architecture",  # 15: cs.AR
            "Computer Vision and Pattern Recognition",  # 16: cs.CV
            "Graphics",  # 17: cs.GR
            "Emerging Technologies",  # 18: cs.ET
            "Systems and Control",  # 19: cs.SY
            "Computational Geometry",  # 20: cs.CG
            "Other Computer Science",  # 21: cs.OH
            "Programming Languages",  # 22: cs.PL
            "Software Engineering",  # 23: cs.SE
            "Machine Learning",  # 24: cs.LG
            "Sound",  # 25: cs.SD
            "Social and Information Networks",  # 26: cs.SI
            "Robotics",  # 27: cs.RO
            "Information Theory",  # 28: cs.IT
            "Performance",  # 29: cs.PF
            "Computation and Language",  # 30: cs.CL
            "Information Retrieval",  # 31: cs.IR
            "Mathematical Software",  # 32: cs.MS
            "Formal Languages and Automata Theory",  # 33: cs.FL
            "Data Structures and Algorithms",  # 34: cs.DS
            "Operating Systems",  # 35: cs.OS
            "Computer Science and Game Theory",  # 36: cs.GT
            "Databases",  # 37: cs.DB
            "Digital Libraries",  # 38: cs.DL
            "Discrete Mathematics",  # 39: cs.DM
        ],
    },
}

# Neighbor sampling config (aligned with LLaGA)
MAX_NEIGHBORS_PER_HOP = 10
MAX_HOPS = 2


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------


def _wrap_user_chat_template(
    content_toks: list[int], tokenizer
) -> tuple[list[int], int]:
    """Splice content tokens into a Llama-3-style user-role chat template.

    Why splice instead of decode→apply_chat_template: decoding tokens to text
    and re-tokenizing is not a bijection — whitespace and special-token
    boundaries can drift, which would shift node_spans relative to input_ids.
    """
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True, add_generation_prompt=True,
    )
    eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
    offset = empty.index(eot_id)
    return list(empty[:offset]) + list(content_toks) + list(empty[offset:]), offset


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


# ---------------------------------------------------------------------------
# Prompt / sample construction
# ---------------------------------------------------------------------------


ANSWER_LABEL_STYLES = ("digit0", "digit0_pad", "number1", "letter")


def get_answer_labels(
    num_classes: int,
    style: str = "digit0",
) -> list[str]:
    """
    Generate answer labels for MC format.

    Styles:
      - digit0:      "0", "1", ..., "N-1"   (single digits 0-9 are 1 token, 10+ are 2 tokens)
      - digit0_pad:  "00", "01", ...        (zero-padded so every label has uniform token length)
      - number1:     "1", "2", ..., "N"
      - letter:      "a", "b", ..., "z", "aa", ...
    """
    if style == "digit0":
        return [str(i) for i in range(num_classes)]
    if style == "digit0_pad":
        width = max(2, len(str(max(num_classes - 1, 0))))
        return [str(i).zfill(width) for i in range(num_classes)]
    if style == "number1":
        return [str(i + 1) for i in range(num_classes)]
    if style == "letter":
        labels = []
        for i in range(num_classes):
            n = i
            chars = []
            while True:
                chars.append(chr(ord("a") + (n % 26)))
                n = n // 26 - 1
                if n < 0:
                    break
            labels.append("".join(reversed(chars)))
        return labels
    raise ValueError(
        f"Unknown answer_label_style={style!r}; expected one of {ANSWER_LABEL_STYLES}"
    )


NEIGHBOR_LABEL_FORMATS = ("bracket", "paren", "sentence", "colon")


def _neighbor_prefix_str(
    idx: int,
    neighbor_labels: Optional[list[int]],
    class_names: list[str],
    answer_labels: Optional[list[str]],
    include_neighbor_labels: bool,
    neighbor_label_format: str = "bracket",
) -> str:
    """Build the neighbor prefix. If include_neighbor_labels is False, returns
    '\\nNeighbor <i>: '. Otherwise returns a label-annotated prefix per
    neighbor_label_format ∈ {bracket, paren, sentence, colon}."""
    if (
        include_neighbor_labels
        and neighbor_labels is not None
        and idx < len(neighbor_labels)
        and neighbor_labels[idx] is not None
    ):
        lbl = neighbor_labels[idx]
        if 0 <= lbl < len(class_names):
            cls = class_names[lbl]
            if answer_labels is not None and lbl < len(answer_labels):
                cls = f"{answer_labels[lbl]}. {cls}"
            n = idx + 1
            if neighbor_label_format == "bracket":
                return f"\nNeighbor {n} [{cls}]: "
            if neighbor_label_format == "paren":
                return f"\nNeighbor {n} (category: {cls}): "
            if neighbor_label_format == "sentence":
                return f"\nNeighbor {n} is a {cls} paper: "
            if neighbor_label_format == "colon":
                return f"\nNeighbor {n} — {cls}: "
            raise ValueError(
                f"Unknown neighbor_label_format={neighbor_label_format!r}; "
                f"expected one of {NEIGHBOR_LABEL_FORMATS}"
            )
    return f"\nNeighbor {idx + 1}: "


def _neighbor_prefix_tokens_and_label_span(
    idx: int,
    neighbor_labels: Optional[list[int]],
    class_names: list[str],
    answer_labels: Optional[list[str]],
    include_neighbor_labels: bool,
    neighbor_label_format: str,
    tok_fn,
) -> tuple[list[int], Optional[tuple[int, int]]]:
    """
    Build neighbor prefix tokens and, when present, return class-label token span.

    Returns:
        prefix_tokens: tokenized neighbor prefix
        label_span: (start, end) token span of the neighbor class label within
            prefix_tokens, or None if no neighbor label is injected.
    """
    if (
        include_neighbor_labels
        and neighbor_labels is not None
        and idx < len(neighbor_labels)
        and neighbor_labels[idx] is not None
    ):
        lbl = neighbor_labels[idx]
        if 0 <= lbl < len(class_names):
            cls = class_names[lbl]
            if answer_labels is not None and lbl < len(answer_labels):
                cls = f"{answer_labels[lbl]}. {cls}"
            n = idx + 1
            if neighbor_label_format == "bracket":
                pre = f"\nNeighbor {n} ["
                post = "]: "
            elif neighbor_label_format == "paren":
                pre = f"\nNeighbor {n} (category: "
                post = "): "
            elif neighbor_label_format == "sentence":
                pre = f"\nNeighbor {n} is a "
                post = " paper: "
            elif neighbor_label_format == "colon":
                pre = f"\nNeighbor {n} — "
                post = ": "
            else:
                raise ValueError(
                    f"Unknown neighbor_label_format={neighbor_label_format!r}; "
                    f"expected one of {NEIGHBOR_LABEL_FORMATS}"
                )

            pre_toks = tok_fn(pre)
            cls_toks = tok_fn(cls)
            post_toks = tok_fn(post)
            start = len(pre_toks)
            end = start + len(cls_toks)
            return pre_toks + cls_toks + post_toks, (start, end)

    prefix = _neighbor_prefix_str(
        idx=idx,
        neighbor_labels=neighbor_labels,
        class_names=class_names,
        answer_labels=answer_labels,
        include_neighbor_labels=include_neighbor_labels,
        neighbor_label_format=neighbor_label_format,
    )
    return tok_fn(prefix), None


def _build_node_sample_chat(
    target_node_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    class_names: list[str],
    tokenizer,
    max_seq_len: int,
    mask_target_text: bool,
    options_str: str,
    answer_tokens: list[int],
    max_answer_tokens: int,
    prompt_layout: str,
    answer_labels: Optional[list[str]] = None,
    neighbor_labels: Optional[list[int]] = None,
    include_neighbor_labels: bool = False,
    neighbor_label_format: str = "bracket",
    mask_neighbor_labels: bool = False,
) -> dict:
    """
    Build a sample wrapped in LLaDA-Instruct chat template.

    Format:
        <|startoftext|><|start_header_id|>user<|end_header_id|>

        {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        {answer_tokens}

    The user_content follows the same layout as the raw prompt
    (target_first or neighbor_first), but wrapped in the chat template.
    Answer tokens are placed in the assistant turn.
    """

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    mask_id = tokenizer.mask_token_id

    # --- Compute chat template overhead ---
    # Tokenize an empty chat template to measure its token count
    empty_template_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    template_overhead = len(empty_template_ids)
    # Budget for content tokens inside the user turn
    content_budget = max_seq_len - template_overhead - len(answer_tokens)

    # --- Build target text portion ---
    target_prefix_str = "Paper: "
    options_suffix_str = f"\nOptions: {options_str}\nAnswer:"

    target_prefix_toks = _tok(target_prefix_str)
    target_body_toks = _tok(target_node_text)
    options_suffix_toks = _tok(options_suffix_str)

    fixed_target_overhead = len(target_prefix_toks) + len(options_suffix_toks)
    target_body_budget = max(content_budget // 2 - fixed_target_overhead, 50)
    target_body_toks = target_body_toks[:target_body_budget]

    target_content_toks = target_prefix_toks + target_body_toks + options_suffix_toks
    target_content_len = len(target_content_toks)

    # --- Build neighbor text portions ---
    num_neighbors = len(neighbor_texts)
    nb_remaining = content_budget - target_content_len
    per_nb_budget = (
        max(nb_remaining // max(num_neighbors, 1), 20) if num_neighbors > 0 else 0
    )

    nb_token_list = []
    for i, (nb_text, hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        nb_prefix, nb_label_span_in_prefix = _neighbor_prefix_tokens_and_label_span(
            idx=i,
            neighbor_labels=neighbor_labels,
            class_names=class_names,
            answer_labels=answer_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            tok_fn=_tok,
        )
        nb_body = _tok(nb_text)
        nb_full = nb_prefix + nb_body
        nb_ids = nb_full[:per_nb_budget]
        if nb_label_span_in_prefix is not None:
            ls, le = nb_label_span_in_prefix
            if ls >= len(nb_ids):
                nb_label_span_in_nb = None
            else:
                nb_label_span_in_nb = (ls, min(le, len(nb_ids)))
        else:
            nb_label_span_in_nb = None
        nb_token_list.append((nb_ids, hop, nb_label_span_in_nb))

    # --- Assemble content tokens (inside user turn) based on layout ---
    if prompt_layout == "neighbor_first":
        content_toks = []
        nb_content_spans = []  # spans relative to content_toks start
        nb_label_content_spans = []  # (start, end, orig_toks_or_None)
        nb_hops = []
        max_nb_total = content_budget - target_content_len

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(content_toks) + len(nb_ids) > max_nb_total:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(content_toks)
            content_toks.extend(nb_ids)
            nb_content_spans.append([start, len(content_toks)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_content_spans.append((start + ls, start + le, orig_toks))
            nb_hops.append(hop)

        target_content_start = len(content_toks)
        content_toks.extend(target_content_toks)
        target_content_span = [target_content_start, len(content_toks)]
    else:
        # target_first (default)
        content_toks = list(target_content_toks)
        target_content_span = [0, target_content_len]
        nb_content_spans = []
        nb_label_content_spans = []  # (start, end, orig_toks_or_None)
        nb_hops = []

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(content_toks) >= content_budget:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(content_toks)
            content_toks.extend(nb_ids)
            nb_content_spans.append([start, len(content_toks)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_content_spans.append((start + ls, start + le, orig_toks))
            nb_hops.append(hop)

    template_ids, content_offset = _wrap_user_chat_template(content_toks, tokenizer)
    input_ids = template_ids + answer_tokens
    label_token_pos = len(template_ids)

    # --- Compute node_spans in final input_ids coordinates ---
    # Shift content spans by content_offset
    target_span = [
        target_content_span[0] + content_offset,
        target_content_span[1] + content_offset,
    ]
    # Include the answer tokens in the target span
    target_span_with_answer = [target_span[0], len(input_ids)]

    node_spans = [target_span_with_answer]
    node_hops_out = [0]

    for span, hop in zip(nb_content_spans, nb_hops):
        node_spans.append([span[0] + content_offset, span[1] + content_offset])
        node_hops_out.append(hop)
    nb_label_abs_spans = [
        (s + content_offset, e + content_offset, orig) for s, e, orig in nb_label_content_spans
    ]

    # Enforce max_seq_len
    input_ids = input_ids[:max_seq_len]

    # --- Labels: -100 everywhere except answer positions ---
    labels = [-100] * len(input_ids)
    if mask_target_text:
        # Target body starts at: content_offset + len(target_prefix_toks)
        # (in neighbor_first: offset by target_content_start)
        if prompt_layout == "neighbor_first":
            tb_start = content_offset + target_content_start + len(target_prefix_toks)
            tb_end = (
                content_offset
                + target_content_start
                + len(target_prefix_toks)
                + len(target_body_toks)
            )
        else:
            tb_start = content_offset + len(target_prefix_toks)
            tb_end = content_offset + len(target_prefix_toks) + len(target_body_toks)
        for pos in range(tb_start, min(tb_end, len(labels))):
            labels[pos] = input_ids[pos]

    # Always include answer tokens in labels
    for j in range(len(answer_tokens)):
        pos = label_token_pos + j
        if pos < len(labels):
            labels[pos] = input_ids[pos]
    for span_start, span_end, orig_toks in nb_label_abs_spans:
        for j, pos in enumerate(range(span_start, min(span_end, len(labels)))):
            labels[pos] = orig_toks[j] if orig_toks is not None else input_ids[pos]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
        "answer_len": len(answer_tokens),
    }


def _build_node_sample_category(
    target_node_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    class_names: list[str],
    tokenizer,
    max_seq_len: int,
    mask_target_text: bool,
    max_answer_tokens: int,
    prompt_layout: str,
    answer_labels: Optional[list[str]] = None,
    neighbor_labels: Optional[list[int]] = None,
    include_neighbor_labels: bool = False,
    neighbor_label_format: str = "bracket",
    include_options: bool = True,
    mask_neighbor_labels: bool = False,
) -> dict:
    """
    Build a sample using natural category infill format.

    Format (include_options=True, default):
        Paper: <target_text>
        Options: 0) Case Based 1) Genetic Algorithms ...
        The category of this paper is: <class_name>

    Format (include_options=False, pure open-ended):
        Paper: <target_text>
        The category of this paper is: <class_name>

    The class name tokens are the answer (masked during eval).
    The reserved fixed window of ``max_answer_tokens`` is filled as
    [real class-name tokens] + [eos_token_id × (max_answer_tokens - answer_len)].
    The entire window is supervised (labels = input_ids on those positions),
    teaching the model to emit EOS after short class names.
    """

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # --- Build answer tokens ---
    # Fixed reserved answer window: real class-name tokens followed by
    # eos_token_id padding. The EOS slots are supervised (labels = eos_id) and
    # remain visible to attention, so the model learns to emit EOS after short
    # class names — matching LLaDA's original SFT recipe (examples/llada/sft.py
    # uses label_pad_token_id = pad_token_id with NoAttentionMaskWrapper).
    class_name = class_names[cls_label]
    answer_tokens = _tok(class_name)[:max_answer_tokens]
    answer_len = len(answer_tokens)
    eos_id = tokenizer.eos_token_id
    assert eos_id is not None, "tokenizer must have eos_token_id"
    while len(answer_tokens) < max_answer_tokens:
        answer_tokens.append(eos_id)

    # --- Tokenize target section ---
    target_prefix = _tok("Paper: ")
    target_body = _tok(target_node_text)
    if include_options:
        options_intro = _tok("\nOptions: ")
        options_body = []
        for i, name in enumerate(class_names):
            option_label = (
                str(i)
                if answer_labels is None or i >= len(answer_labels)
                else answer_labels[i]
            )
            option_prefix = _tok(f"{'' if i == 0 else ' '}{option_label}. ")
            options_body.extend(option_prefix)
            options_body.extend(_tok(name))
        options_suffix = options_intro + options_body
    else:
        options_suffix = []
    category_suffix = _tok("\nThe category of this paper is: ")

    # Reserve space: target gets up to half of max_seq_len
    overhead = (
        len(target_prefix)
        + len(options_suffix)
        + len(category_suffix)
        + len(answer_tokens)
    )
    target_body_budget = max(max_seq_len // 2 - overhead, 50)
    target_body = target_body[:target_body_budget]

    # Target section as a unit (includes options + suffix + answer)
    target_section = (
        target_prefix + target_body + options_suffix + category_suffix + answer_tokens
    )
    target_section_len = len(target_section)
    answer_offset_in_target = (
        len(target_prefix)
        + len(target_body)
        + len(options_suffix)
        + len(category_suffix)
    )
    # --- Tokenize neighbor sections ---
    num_neighbors = len(neighbor_texts)
    remaining_budget = max_seq_len - target_section_len
    per_nb_budget = (
        max(remaining_budget // max(num_neighbors, 1), 20) if num_neighbors > 0 else 0
    )

    nb_token_list = []
    for i, (nb_text, hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        nb_prefix, nb_label_span_in_prefix = _neighbor_prefix_tokens_and_label_span(
            idx=i,
            neighbor_labels=neighbor_labels,
            class_names=class_names,
            answer_labels=answer_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            tok_fn=_tok,
        )
        nb_body = _tok(nb_text)
        nb_full = nb_prefix + nb_body
        nb_ids = nb_full[:per_nb_budget]
        if nb_label_span_in_prefix is not None:
            ls, le = nb_label_span_in_prefix
            if ls >= len(nb_ids):
                nb_label_span_in_nb = None
            else:
                nb_label_span_in_nb = (ls, min(le, len(nb_ids)))
        else:
            nb_label_span_in_nb = None
        nb_token_list.append((nb_ids, hop, nb_label_span_in_nb))

    # --- Assemble based on layout ---
    # For category_infill, answer is always at the end of target section.
    # target_first: [target+answer] [nb1] [nb2] ...
    # neighbor_first: [nb1] [nb2] ... [target+answer]
    if prompt_layout == "neighbor_first":
        input_ids = []
        nb_spans = []
        nb_label_spans_abs = []  # (start, end, orig_toks_or_None)
        nb_hops = []
        max_nb_total = max_seq_len - target_section_len

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(input_ids) + len(nb_ids) > max_nb_total:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(input_ids)
            input_ids.extend(nb_ids)
            nb_spans.append([start, len(input_ids)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_spans_abs.append((start + ls, start + le, orig_toks))
            nb_hops.append(hop)

        target_start = len(input_ids)
        label_token_pos = target_start + answer_offset_in_target
        input_ids.extend(target_section)

        node_spans = [[target_start, len(input_ids)]] + nb_spans
        node_hops_out = [0] + nb_hops
    else:
        # target_first (default)
        input_ids = list(target_section)
        label_token_pos = answer_offset_in_target
        node_spans = [[0, target_section_len]]
        node_hops_out = [0]
        nb_label_spans_abs = []  # (start, end, orig_toks_or_None)

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(input_ids) >= max_seq_len:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(input_ids)
            input_ids.extend(nb_ids)
            node_spans.append([start, len(input_ids)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_spans_abs.append((start + ls, start + le, orig_toks))
            node_hops_out.append(hop)

    # Enforce max_seq_len
    input_ids = input_ids[:max_seq_len]

    # --- Labels: -100 everywhere except answer positions ---
    labels = [-100] * len(input_ids)
    if mask_target_text:
        if prompt_layout == "neighbor_first":
            tb_start = target_start + len(target_prefix)
            tb_end = target_start + len(target_prefix) + len(target_body)
        else:
            tb_start = len(target_prefix)
            tb_end = len(target_prefix) + len(target_body)
        for pos in range(tb_start, min(tb_end, len(labels))):
            labels[pos] = input_ids[pos]
    # Answer tokens in labels: supervise the entire fixed window
    # (real class-name tokens + trailing EOS slots). EOS supervision teaches
    # the model to terminate after short class names; this matches LLaDA's
    # original SFT recipe.
    for j in range(max_answer_tokens):
        pos = label_token_pos + j
        if pos < len(labels):
            labels[pos] = input_ids[pos]
    for span_start, span_end, orig_toks in nb_label_spans_abs:
        for j, pos in enumerate(range(span_start, min(span_end, len(labels)))):
            labels[pos] = orig_toks[j] if orig_toks is not None else input_ids[pos]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
        "answer_len": answer_len,
    }


def _build_node_sample_nd(
    target_title: str,
    target_abstract: Optional[str],
    target_label_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    tokenizer,
    max_seq_len: int,
    include_abstract: bool = False,
) -> dict:
    """LLaGA-style node description sample (Option A1).

    Center text is excluded from the prompt; the model produces an LLaGA-style
    description from neighbors alone. Labels are -100 except on the assistant
    span, so the TM-DLM trainer applies diffusion masking + loss only there.
    """
    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # Assistant target (LLaGA template). `nda_describe` adds the abstract.
    title = (target_title or "").strip()
    if include_abstract:
        ab = (target_abstract or "").strip()
        parts = [f"This is a paper in {target_label_text} domain"]
        if title: parts.append(f"its title is {title}")
        if ab: parts.append(f"its abstract is {ab}")
        assistant_text = ", ".join(parts) + "."
    elif title:
        assistant_text = f"This is a paper in {target_label_text} domain, it's about {title}."
    else:
        assistant_text = f"This is a paper in {target_label_text} domain."
    answer_tokens = _tok(assistant_text)

    empty_prefix_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], tokenize=True, add_generation_prompt=True,
    )
    instruction_toks = _tok("\nPlease briefly describe the center paper.")
    content_budget = max_seq_len - len(empty_prefix_ids) - len(answer_tokens)
    nb_budget_total = max(content_budget - len(instruction_toks), 0)
    per_nb_budget = (
        max(nb_budget_total // max(len(neighbor_texts), 1), 20) if neighbor_texts else 0
    )

    content_toks: list[int] = []
    nb_spans: list[list[int]] = []
    nb_hops_kept: list[int] = []
    for i, (text, hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        nb_ids = _tok(f"\nNeighbor {i + 1}: {text}")[:per_nb_budget]
        if len(content_toks) + len(nb_ids) > nb_budget_total:
            break
        start = len(content_toks)
        content_toks.extend(nb_ids)
        nb_spans.append([start, len(content_toks)])
        nb_hops_kept.append(hop)
    content_toks.extend(instruction_toks)

    template_ids, content_offset = _wrap_user_chat_template(content_toks, tokenizer)
    # Protect the answer span: if template overflows, trim template, never answer.
    max_template_len = max(max_seq_len - len(answer_tokens), 0)
    if len(template_ids) > max_template_len:
        template_ids = template_ids[:max_template_len]
    input_ids = (template_ids + answer_tokens)[:max_seq_len]
    label_token_pos = len(template_ids)

    labels = [-100] * len(input_ids)
    for j in range(len(answer_tokens)):
        pos = label_token_pos + j
        if pos < len(input_ids):
            labels[pos] = input_ids[pos]

    node_spans = [[label_token_pos, len(input_ids)]]
    node_hops_out = [0]
    for (s, e), hop in zip(nb_spans, nb_hops_kept):
        node_spans.append([s + content_offset, e + content_offset])
        node_hops_out.append(hop)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
        "answer_len": len(answer_tokens),
    }


def build_node_sample(
    target_node_text: str,
    neighbor_texts: list[str],
    neighbor_hops: list[int],
    cls_label: int,
    class_names: list[str],
    tokenizer,
    max_seq_len: int = 2048,
    mask_target_text: bool = False,
    answer_labels: Optional[list[str]] = None,
    max_answer_tokens: int = 1,
    prompt_layout: str = "target_first",
    use_chat_template: bool = False,
    prompt_format: str = "mc_digit",
    answer_label_style: str = "digit0",
    neighbor_labels: Optional[list[int]] = None,
    include_neighbor_labels: bool = False,
    neighbor_label_format: str = "bracket",
    include_options: bool = True,
    mask_neighbor_labels: bool = False,
    target_title: Optional[str] = None,
    target_abstract: Optional[str] = None,
    target_label_text: Optional[str] = None,
) -> dict:
    """
    Build a single TM-DLM training sample for one target node.

    ``prompt_format`` controls the overall format:

    mc_digit (default):
        Multiple-choice with digit answers.
        Paper: <target_text>
        Options: 0) Case Based ... 6) Theory
        Answer: <answer_digit>

    category_infill:
        Natural language category infill with options list.
        Paper: <target_text>
        Options: 0) Case Based 1) Genetic Algorithms ...
        The category of this paper is: <class_name>

    nd_describe / nda_describe:
        LLaGA-style node description. Center node text is NOT in the prompt;
        only neighbor texts and an instruction are. The model must produce
        a templated description. ``nda_describe`` includes the abstract in
        the answer; ``nd_describe`` uses only the title.
        Requires ``target_title`` and ``target_label_text``; ``target_abstract``
        is required for ``nda_describe``.

    ``prompt_layout`` controls ordering (target_first vs neighbor_first).

    Labels are -100 everywhere EXCEPT at the answer token positions.
    """

    if prompt_format in ("nd_describe", "nda_describe"):
        return _build_node_sample_nd(
            target_title=target_title,
            target_abstract=target_abstract,
            target_label_text=target_label_text,
            neighbor_texts=neighbor_texts,
            neighbor_hops=neighbor_hops,
            cls_label=cls_label,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            include_abstract=(prompt_format == "nda_describe"),
        )

    if prompt_format == "category_infill":
        return _build_node_sample_category(
            target_node_text=target_node_text,
            neighbor_texts=neighbor_texts,
            neighbor_hops=neighbor_hops,
            cls_label=cls_label,
            class_names=class_names,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_target_text=mask_target_text,
            max_answer_tokens=max_answer_tokens,
            prompt_layout=prompt_layout,
            answer_labels=answer_labels,
            neighbor_labels=neighbor_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            include_options=include_options,
            mask_neighbor_labels=mask_neighbor_labels,
        )

    # --- mc_digit format (original) ---

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    mask_id = tokenizer.mask_token_id

    # --- Build options string ---
    if answer_labels is None:
        answer_labels = get_answer_labels(
            len(class_names),
            style=answer_label_style,
        )
    options_str = " ".join(
        f"{answer_labels[i]}) {name}" for i, name in enumerate(class_names)
    )

    # --- Build answer tokens ---
    answer_str = answer_labels[cls_label]
    answer_tokens = _tok(answer_str)[:max_answer_tokens]

    if use_chat_template:
        return _build_node_sample_chat(
            target_node_text=target_node_text,
            neighbor_texts=neighbor_texts,
            neighbor_hops=neighbor_hops,
            cls_label=cls_label,
            class_names=class_names,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_target_text=mask_target_text,
            options_str=options_str,
            answer_tokens=answer_tokens,
            max_answer_tokens=max_answer_tokens,
            prompt_layout=prompt_layout,
            answer_labels=answer_labels,
            neighbor_labels=neighbor_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            mask_neighbor_labels=mask_neighbor_labels,
        )

    # --- Tokenize target section ---
    target_prefix = _tok("Paper: ")
    target_body = _tok(target_node_text)
    options_intro = _tok("\nOptions: ")
    options_body = []
    for i, name in enumerate(class_names):
        opt_prefix = _tok(f"{'' if i == 0 else ' '}{answer_labels[i]}) ")
        options_body.extend(opt_prefix)
        options_body.extend(_tok(name))
    options_suffix = _tok("\nAnswer: ")
    options_prefix = options_intro + options_body + options_suffix

    # Reserve space: target gets up to half of max_seq_len
    overhead = len(target_prefix) + len(options_prefix) + len(answer_tokens)
    target_body_budget = max(max_seq_len // 2 - overhead, 50)
    target_body = target_body[:target_body_budget]

    # Target section as a unit
    target_section = target_prefix + target_body + options_prefix + answer_tokens
    target_section_len = len(target_section)
    answer_offset_in_target = (
        len(target_prefix) + len(target_body) + len(options_prefix)
    )

    # --- Tokenize neighbor sections ---
    num_neighbors = len(neighbor_texts)
    remaining_budget = max_seq_len - target_section_len
    per_nb_budget = (
        max(remaining_budget // max(num_neighbors, 1), 20) if num_neighbors > 0 else 0
    )

    nb_token_list = []
    for i, (nb_text, hop) in enumerate(zip(neighbor_texts, neighbor_hops)):
        nb_prefix, nb_label_span_in_prefix = _neighbor_prefix_tokens_and_label_span(
            idx=i,
            neighbor_labels=neighbor_labels,
            class_names=class_names,
            answer_labels=answer_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            tok_fn=_tok,
        )
        nb_body = _tok(nb_text)
        nb_full = nb_prefix + nb_body
        nb_ids = nb_full[:per_nb_budget]
        if nb_label_span_in_prefix is not None:
            ls, le = nb_label_span_in_prefix
            if ls >= len(nb_ids):
                nb_label_span_in_nb = None
            else:
                nb_label_span_in_nb = (ls, min(le, len(nb_ids)))
        else:
            nb_label_span_in_nb = None
        nb_token_list.append((nb_ids, hop, nb_label_span_in_nb))

    # --- Assemble based on layout ---
    if prompt_layout == "neighbor_first":
        input_ids = []
        nb_spans = []
        nb_label_spans_abs = []  # (start, end, orig_toks_or_None)
        nb_hops = []
        max_nb_total = max_seq_len - target_section_len

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(input_ids) + len(nb_ids) > max_nb_total:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(input_ids)
            input_ids.extend(nb_ids)
            nb_spans.append([start, len(input_ids)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_spans_abs.append((start + ls, start + le, orig_toks))
            nb_hops.append(hop)

        # Append target section after neighbors
        target_start = len(input_ids)
        label_token_pos = target_start + answer_offset_in_target
        input_ids.extend(target_section)

        # Target is always first in node_spans (hop=0)
        node_spans = [[target_start, len(input_ids)]] + nb_spans
        node_hops_out = [0] + nb_hops
    else:
        # "target_first" (default)
        input_ids = list(target_section)
        label_token_pos = answer_offset_in_target
        node_spans = [[0, target_section_len]]
        node_hops_out = [0]
        nb_label_spans_abs = []  # (start, end, orig_toks_or_None)

        for nb_ids, hop, nb_label_span in nb_token_list:
            if len(input_ids) >= max_seq_len:
                break
            nb_ids = list(nb_ids)
            orig_toks = None
            if mask_neighbor_labels and nb_label_span is not None and mask_id is not None:
                ls, le = nb_label_span
                orig_toks = nb_ids[ls:le]
                nb_ids[ls:le] = [mask_id] * (le - ls)
            start = len(input_ids)
            input_ids.extend(nb_ids)
            node_spans.append([start, len(input_ids)])
            if nb_label_span is not None:
                ls, le = nb_label_span
                nb_label_spans_abs.append((start + ls, start + le, orig_toks))
            node_hops_out.append(hop)

    # Enforce max_seq_len
    input_ids = input_ids[:max_seq_len]

    # --- Labels: -100 everywhere except masked positions ---
    labels = [-100] * len(input_ids)
    if mask_target_text:
        if prompt_layout == "neighbor_first":
            target_body_start = target_start + len(target_prefix)
            target_body_end = target_start + len(target_prefix) + len(target_body)
        else:
            target_body_start = len(target_prefix)
            target_body_end = len(target_prefix) + len(target_body)
        for pos in range(target_body_start, min(target_body_end, len(labels))):
            labels[pos] = input_ids[pos]
    # Supervise the answer digit only. Do NOT supervise option-block tokens:
    # eval masks all `labels != -100` positions, which would otherwise turn the
    # gold option's class name into the only [MASK] span and leak the answer.
    for j in range(len(answer_tokens)):
        pos = label_token_pos + j
        if pos < len(labels):
            labels[pos] = input_ids[pos]
    for span_start, span_end, orig_toks in nb_label_spans_abs:
        for j, pos in enumerate(range(span_start, min(span_end, len(labels)))):
            labels[pos] = orig_toks[j] if orig_toks is not None else input_ids[pos]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops_out,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
        "answer_len": len(answer_tokens),
    }


def _build_tag_samples(
    split_ids: list[int],
    node_data: dict[int, dict],
    adj: dict[int, list[int]],
    class_names: list[str],
    tokenizer,
    max_seq_len: int,
    max_neighbors_per_hop: int,
    max_hops: int,
    seed: int,
    mask_target_text: bool,
    max_answer_tokens: int,
    prompt_layout: str,
    use_chat_template: bool = False,
    prompt_format: str = "mc_digit",
    answer_label_style: str = "digit0",
    include_neighbor_labels: bool = False,
    neighbor_label_format: str = "bracket",
    include_options: bool = True,
    mask_neighbor_labels: bool = False,
    neighbor_seed: int | None = None,
) -> list[dict]:
    """Build TM-DLM samples for a list of node IDs (shared across all datasets).

    neighbor_seed: if set, governs neighbor sampling RNG independently of `seed`.
    Used at eval-time to do TTA with fixed sample selection but jittered neighbors.
    """
    answer_labels = get_answer_labels(
        len(class_names),
        style=answer_label_style,
    )
    rng = random.Random(seed if neighbor_seed is None else neighbor_seed)
    samples = []

    for node_id in split_ids:
        if node_id not in node_data:
            continue
        info = node_data[node_id]

        nb_ids, nb_hops = _sample_khop_neighbors(
            adj, node_id, max_neighbors_per_hop, max_hops, rng
        )

        neighbor_texts = []
        neighbor_hops = []
        neighbor_labels = []
        for nb_id, hop in zip(nb_ids, nb_hops):
            if nb_id in node_data:
                nd = node_data[nb_id]
                neighbor_texts.append(f"{nd['title']}. {nd['abstract']}")
                neighbor_hops.append(hop)
                neighbor_labels.append(nd.get("label"))

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
            answer_labels=answer_labels,
            max_answer_tokens=max_answer_tokens,
            prompt_layout=prompt_layout,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
            answer_label_style=answer_label_style,
            neighbor_labels=neighbor_labels,
            include_neighbor_labels=include_neighbor_labels,
            neighbor_label_format=neighbor_label_format,
            include_options=include_options,
            mask_neighbor_labels=mask_neighbor_labels,
            target_title=info.get("title"),
            target_abstract=info.get("abstract"),
            target_label_text=class_names[info["label"]],
        )
        samples.append(result)

    return samples



def _tag_dataset_cache_key(
    *,
    dataset_name,
    tokenizer,
    split,
    max_seq_len,
    max_neighbors_per_hop,
    max_hops,
    seed,
    mask_target_text,
    max_answer_tokens,
    prompt_layout,
    use_chat_template,
    prompt_format,
    answer_label_style,
    include_neighbor_labels,
    neighbor_label_format,
    include_options,
    mask_neighbor_labels,
    max_samples,
    balance_merged,
    resample_strategy,
    boost_spec,
    neighbor_seed=None,
) -> str:
    """Hash all args that affect sample content into a 16-hex cache key."""
    if isinstance(dataset_name, (list, tuple)):
        ds_repr = "+".join(sorted(str(d) for d in dataset_name))
    else:
        ds_repr = str(dataset_name)
    payload = {
        "version": _TAG_CACHE_VERSION,
        "dataset_name": ds_repr,
        "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
        "vocab_size": len(tokenizer),
        "split": split,
        "max_seq_len": int(max_seq_len),
        "max_neighbors_per_hop": int(max_neighbors_per_hop),
        "max_hops": int(max_hops),
        "seed": int(seed),
        "mask_target_text": bool(mask_target_text),
        "max_answer_tokens": int(max_answer_tokens),
        "prompt_layout": prompt_layout,
        "use_chat_template": bool(use_chat_template),
        "prompt_format": prompt_format,
        "answer_label_style": answer_label_style,
        "include_neighbor_labels": bool(include_neighbor_labels),
        "neighbor_label_format": neighbor_label_format,
        "include_options": bool(include_options),
        "mask_neighbor_labels": bool(mask_neighbor_labels),
        "max_samples": int(max_samples or 0),
        "balance_merged": bool(balance_merged),
        "resample_strategy": resample_strategy,
        "boost_spec": boost_spec,
    }
    if neighbor_seed is not None:
        payload["neighbor_seed"] = int(neighbor_seed)
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _tag_save_to_cache(dataset: Dataset, cache_dir: Path) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # save_to_disk requires an empty / new dir
        if (cache_dir / "dataset_info.json").exists():
            return
        dataset.save_to_disk(str(cache_dir))
        logger.info("[tag-cache] saved → %s (%d samples)", cache_dir, len(dataset))
    except Exception as e:  # pragma: no cover - non-fatal cache miss
        logger.warning("[tag-cache] save failed for %s: %s", cache_dir, e)


def load_tag_dataset(
    dataset_name,
    tokenizer,
    split: str = "train",
    max_seq_len: int = 2048,
    max_neighbors_per_hop: int = MAX_NEIGHBORS_PER_HOP,
    max_hops: int = MAX_HOPS,
    seed: int = 42,
    mask_target_text: bool = False,
    max_answer_tokens: int = 1,
    prompt_layout: str = "target_first",
    use_chat_template: bool = False,
    prompt_format: str = "mc_digit",
    answer_label_style: str = "digit0",
    include_neighbor_labels: bool = False,
    neighbor_label_format: str = "bracket",
    include_options: bool = True,
    mask_neighbor_labels: bool = False,
    max_samples: int = 0,
    balance_merged: bool = False,
    resample_strategy: str = "none",
    boost_spec: str = "",
    neighbor_seed: int | None = None,
) -> Dataset:
    """
    Load a TAG dataset and return a HuggingFace Dataset of TM-DLM samples.

    Supports: "cora", "pubmed", "ogbn-arxiv".

    `dataset_name` accepts either a single dataset name (str) or a list of
    names (e.g. ["cora", "pubmed"]) to **merge** datasets. When merging,
    each sample carries its own dataset's class list inside its prompt, so
    the answer space is per-sample and class indices do not collide.
    Samples are concatenated and shuffled with the provided seed; if
    `max_samples > 0` it caps total merged size (sampled proportionally).

    When merging, `balance_merged=True` downsamples each dataset to
    `min(per-dataset sample counts)` before concatenation, so every dataset
    contributes the same number of samples. Useful when one dataset is much
    larger and would otherwise dominate the gradient.

    On-disk cache: built samples are hashed (all args that affect content)
    and persisted under ``TAG_CACHE_ROOT``. Subsequent calls with the same
    args reload in seconds via ``Dataset.load_from_disk``. This is what lets
    DDP rank 0's "dataset prep" finish well within the NCCL bootstrap
    timeout window, instead of re-tokenizing every sample on each launch.
    """
    cache_key = _tag_dataset_cache_key(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split=split,
        max_seq_len=max_seq_len,
        max_neighbors_per_hop=max_neighbors_per_hop,
        max_hops=max_hops,
        seed=seed,
        mask_target_text=mask_target_text,
        max_answer_tokens=max_answer_tokens,
        prompt_layout=prompt_layout,
        use_chat_template=use_chat_template,
        prompt_format=prompt_format,
        answer_label_style=answer_label_style,
        include_neighbor_labels=include_neighbor_labels,
        neighbor_label_format=neighbor_label_format,
        include_options=include_options,
        mask_neighbor_labels=mask_neighbor_labels,
        max_samples=max_samples,
        balance_merged=balance_merged,
        resample_strategy=resample_strategy,
        boost_spec=boost_spec,
        neighbor_seed=neighbor_seed,
    )
    cache_dir = TAG_CACHE_ROOT / cache_key
    if (cache_dir / "dataset_info.json").exists():
        try:
            ds = Dataset.load_from_disk(str(cache_dir))
            logger.info(
                "[tag-cache] hit %s (%d samples) ← %s",
                cache_key, len(ds), cache_dir,
            )
            return ds
        except Exception as e:  # pragma: no cover - cache corruption fallback
            logger.warning(
                "[tag-cache] load failed for %s (%s); rebuilding",
                cache_dir, e,
            )

    if isinstance(dataset_name, (list, tuple)):
        names = list(dataset_name)
        if not names:
            raise ValueError("dataset_name list must be non-empty")
        if len(names) == 1:
            return load_tag_dataset(
                names[0], tokenizer, split=split, max_seq_len=max_seq_len,
                max_neighbors_per_hop=max_neighbors_per_hop, max_hops=max_hops,
                seed=seed, mask_target_text=mask_target_text,
                max_answer_tokens=max_answer_tokens, prompt_layout=prompt_layout,
                use_chat_template=use_chat_template, prompt_format=prompt_format,
                answer_label_style=answer_label_style,
                include_neighbor_labels=include_neighbor_labels,
                neighbor_label_format=neighbor_label_format,
                include_options=include_options,
                mask_neighbor_labels=mask_neighbor_labels,
                max_samples=max_samples,
                balance_merged=balance_merged,
                resample_strategy="none",  # single dataset: skip merge-level strategies
                boost_spec="",
            )
        for n in names:
            if n not in DATASET_CONFIGS:
                raise ValueError(
                    f"Unknown dataset: {n}. Supported: {list(DATASET_CONFIGS.keys())}"
                )

        per_dataset_samples: Dict[str, list] = {}
        class_names_per_dataset: Dict[str, list] = {}
        for n in names:
            node_data, adj, class_names, split_ids = _DATA_LOADERS[n](
                DATASET_CONFIGS[n], split, seed
            )
            ds_samples = _build_tag_samples(
                split_ids, node_data, adj, class_names, tokenizer,
                max_seq_len, max_neighbors_per_hop, max_hops, seed,
                mask_target_text, max_answer_tokens, prompt_layout,
                use_chat_template, prompt_format, answer_label_style,
                include_neighbor_labels=include_neighbor_labels,
                neighbor_label_format=neighbor_label_format,
                include_options=include_options,
                mask_neighbor_labels=mask_neighbor_labels,
            )
            for s in ds_samples:
                s["dataset"] = n
            per_dataset_samples[n] = ds_samples
            class_names_per_dataset[n] = class_names

        # Resampling: prefer the new strategy API; fall back to the legacy
        # `balance_merged=True` flag for backward compatibility.
        effective_strategy = resample_strategy
        if balance_merged and resample_strategy == "none":
            effective_strategy = "balance_datasets"
        if effective_strategy != "none":
            from dllm.data.resample import apply_resampling
            per_dataset_samples = apply_resampling(
                per_dataset_samples,
                strategy=effective_strategy,
                seed=seed,
                boost_spec=boost_spec,
                class_names_per_dataset=class_names_per_dataset,
            )

        per_dataset_counts = {n: len(s) for n, s in per_dataset_samples.items()}

        all_samples = []
        for n in names:
            all_samples.extend(per_dataset_samples[n])

        rng = random.Random(seed)
        rng.shuffle(all_samples)

        if max_samples and max_samples > 0:
            all_samples = all_samples[:max_samples]

        dataset = Dataset.from_list(all_samples)
        dataset.info.description = (
            f"TM-DLM merged ({split}) " + ", ".join(
                f"{n}={per_dataset_counts[n]}" for n in names
            )
        )
        _tag_save_to_cache(dataset, cache_dir)
        return dataset

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}"
        )

    node_data, adj, class_names, split_ids = _DATA_LOADERS[dataset_name](
        DATASET_CONFIGS[dataset_name], split, seed
    )

    if max_samples and max_samples > 0:
        rng = random.Random(seed)
        split_ids = rng.sample(list(split_ids), min(max_samples, len(split_ids)))

    samples = _build_tag_samples(
        split_ids,
        node_data,
        adj,
        class_names,
        tokenizer,
        max_seq_len,
        max_neighbors_per_hop,
        max_hops,
        seed,
        mask_target_text,
        max_answer_tokens,
        prompt_layout,
        use_chat_template,
        prompt_format,
        answer_label_style,
        include_neighbor_labels=include_neighbor_labels,
        neighbor_label_format=neighbor_label_format,
        include_options=include_options,
        mask_neighbor_labels=mask_neighbor_labels,
        neighbor_seed=neighbor_seed,
    )

    # Apply class-level resampling for single-dataset path (boost / balance_classes).
    if resample_strategy in ("balance_classes", "boost"):
        for s in samples:
            s["dataset"] = dataset_name
        from dllm.data.resample import apply_resampling
        out = apply_resampling(
            {dataset_name: samples},
            strategy=resample_strategy,
            seed=seed,
            boost_spec=boost_spec,
            class_names_per_dataset={dataset_name: class_names},
        )
        samples = out[dataset_name]
        rng = random.Random(seed)
        rng.shuffle(samples)

    dataset = Dataset.from_list(samples)
    dataset.info.description = (
        f"TM-DLM {dataset_name} ({split}), {len(class_names)} classes"
    )
    _tag_save_to_cache(dataset, cache_dir)
    return dataset


def get_class_token_ids(
    dataset_name: str,
    tokenizer,
    max_answer_tokens: int = 1,
    prompt_format: str = "mc_digit",
    answer_label_style: str = "digit0",
) -> tuple[list[str], list]:
    """
    Get the answer token IDs for classification.

    prompt_format="mc_digit":
        Uses numeric labels ("0", "1", ...).
        max_answer_tokens=1: returns list[int] of single token IDs.
        max_answer_tokens>1: returns list[list[int]] padded to max_answer_tokens.

    prompt_format="category_infill":
        Uses class name tokens directly.
        Always returns list[list[int]] padded to max_answer_tokens.

    Returns:
        class_names: list of class name strings
        answer_token_ids: token IDs per class
    """
    config = DATASET_CONFIGS[dataset_name]
    num_classes = config["num_classes"]

    llaga_loaded = _load_llaga_processed_data(config, "train")
    if llaga_loaded is not None:
        _, _, class_names, _ = llaga_loaded
    elif "class_names" in config:
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

    if prompt_format == "category_infill":
        # Auto-compute max_answer_tokens from actual class name lengths
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        all_tokens = [
            tokenizer.encode(name, add_special_tokens=False) for name in class_names
        ]
        max_answer_tokens = max(len(t) for t in all_tokens)
        answer_token_ids = []
        for tokens in all_tokens:
            tokens = tokens[:max_answer_tokens]
            tokens = tokens + [pad_id] * (max_answer_tokens - len(tokens))
            answer_token_ids.append(tokens)
        return class_names, answer_token_ids

    # --- mc_digit format ---
    answer_labels = get_answer_labels(
        num_classes,
        style=answer_label_style,
    )

    if max_answer_tokens == 1:
        answer_token_ids = []
        for label in answer_labels:
            tokens = tokenizer.encode(label, add_special_tokens=False)
            answer_token_ids.append(tokens[0])
        return class_names, answer_token_ids

    answer_token_ids = []
    for label in answer_labels:
        tokens = tokenizer.encode(label, add_special_tokens=False)[:max_answer_tokens]
        tokens = tokens + [0] * (max_answer_tokens - len(tokens))
        answer_token_ids.append(tokens)
    return class_names, answer_token_ids


# ---------------------------------------------------------------------------
# Link-prediction sample construction (yes/no answer over two node-centred
# subgraphs in a single sequence; cross-star topology mask).
# ---------------------------------------------------------------------------


# Role tags written into ``node_roles`` and consumed by
# ``GraphDataCollator._build_topology_mask`` to build the LP crossed-star mask.
LP_ROLE_TARGET = 0   # u-text / v-text spans (both are "centres")
LP_ROLE_NBR_U = 1    # u-side neighbour
LP_ROLE_NBR_V = 2    # v-side neighbour
LP_ROLE_QUESTION = 3  # trailing "Connected? Answer: <yes|no>" span


def get_lp_yesno_token_ids(tokenizer) -> list[int]:
    """Return single-token ids for the LP answer set in canonical order
    [no_id, yes_id] so cls_label=0/1 maps to the correct token."""
    no_id = tokenizer.encode(" no", add_special_tokens=False)
    yes_id = tokenizer.encode(" yes", add_special_tokens=False)
    if len(no_id) != 1 or len(yes_id) != 1:
        raise RuntimeError(
            f"Expected single-token ' no' and ' yes', got no={no_id}, yes={yes_id}"
        )
    return [no_id[0], yes_id[0]]


def build_edge_sample(
    u_text: str,
    v_text: str,
    u_neighbor_texts: list[str],
    u_neighbor_hops: list[int],
    v_neighbor_texts: list[str],
    v_neighbor_hops: list[int],
    cls_label: int,
    tokenizer,
    max_seq_len: int = 2048,
    mask_target_text: bool = False,
) -> dict:
    """Build a single LP sample. ``cls_label`` ∈ {0=no, 1=yes}.

    Sequence layout (all tokens are added without special tokens):

        Paper A: <u_text>
        Neighbor A1: <nb_text> ... Neighbor Ak: <nb_text>
        Paper B: <v_text>
        Neighbor B1: <nb_text> ... Neighbor Bk: <nb_text>
        Do Paper A and Paper B cite each other? Answer: <yes|no>

    The trailing answer is one token (' yes' / ' no').

    The returned ``node_spans`` enumerate, in this order:
        [u_text_span, u_nbr_1_span, ..., v_text_span, v_nbr_1_span, ..., q_span]
    with the parallel ``node_roles`` list giving each span's LP role tag.
    """

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    yesno_ids = get_lp_yesno_token_ids(tokenizer)
    answer_token = yesno_ids[cls_label]

    # Question span comes last, before the answer token.
    q_str = "\nDo Paper A and Paper B cite each other? Answer:"
    q_toks = _tok(q_str)

    # Per-section static prefixes.
    paper_a_prefix = _tok("Paper A: ")
    paper_b_prefix = _tok("\nPaper B: ")

    paper_a_body = _tok(u_text)
    paper_b_body = _tok(v_text)

    # Budget plan: split the leftover after fixed overhead into 4 quarters
    # (u_text, u_nbrs, v_text, v_nbrs); per-neighbour budget then shares
    # the per-side neighbour budget evenly.
    fixed_overhead = (
        len(paper_a_prefix) + len(paper_b_prefix) + len(q_toks) + 1  # +1 for answer
    )
    free = max(max_seq_len - fixed_overhead, 200)
    per_quarter = free // 4
    u_text_budget = max(per_quarter, 50)
    v_text_budget = max(per_quarter, 50)
    paper_a_body = paper_a_body[:u_text_budget]
    paper_b_body = paper_b_body[:v_text_budget]

    n_u_nb = len(u_neighbor_texts)
    n_v_nb = len(v_neighbor_texts)
    u_nb_budget_total = per_quarter
    v_nb_budget_total = per_quarter
    per_u_nb = max(u_nb_budget_total // max(n_u_nb, 1), 20) if n_u_nb else 0
    per_v_nb = max(v_nb_budget_total // max(n_v_nb, 1), 20) if n_v_nb else 0

    # ---- Build u side ----
    input_ids: list[int] = []
    node_spans: list[list[int]] = []
    node_roles: list[int] = []
    node_hops: list[int] = []

    u_text_start = len(input_ids)
    input_ids.extend(paper_a_prefix + paper_a_body)
    u_text_end = len(input_ids)
    node_spans.append([u_text_start, u_text_end])
    node_roles.append(LP_ROLE_TARGET)
    node_hops.append(0)

    for idx, (nb_text, hop) in enumerate(zip(u_neighbor_texts, u_neighbor_hops)):
        if len(input_ids) >= max_seq_len - len(q_toks) - 1:
            break
        nb_prefix = _tok(f"\nNeighbor A{idx + 1}: ")
        nb_body = _tok(nb_text)
        nb_full = (nb_prefix + nb_body)[:per_u_nb]
        if not nb_full:
            continue
        s = len(input_ids)
        input_ids.extend(nb_full)
        node_spans.append([s, len(input_ids)])
        node_roles.append(LP_ROLE_NBR_U)
        node_hops.append(hop)

    # ---- Build v side ----
    v_text_start = len(input_ids)
    input_ids.extend(paper_b_prefix + paper_b_body)
    v_text_end = len(input_ids)
    node_spans.append([v_text_start, v_text_end])
    node_roles.append(LP_ROLE_TARGET)
    node_hops.append(0)

    for idx, (nb_text, hop) in enumerate(zip(v_neighbor_texts, v_neighbor_hops)):
        if len(input_ids) >= max_seq_len - len(q_toks) - 1:
            break
        nb_prefix = _tok(f"\nNeighbor B{idx + 1}: ")
        nb_body = _tok(nb_text)
        nb_full = (nb_prefix + nb_body)[:per_v_nb]
        if not nb_full:
            continue
        s = len(input_ids)
        input_ids.extend(nb_full)
        node_spans.append([s, len(input_ids)])
        node_roles.append(LP_ROLE_NBR_V)
        node_hops.append(hop)

    # ---- Question + answer ----
    q_start = len(input_ids)
    input_ids.extend(q_toks)
    label_token_pos = len(input_ids)
    input_ids.append(answer_token)
    q_end = len(input_ids)
    node_spans.append([q_start, q_end])
    node_roles.append(LP_ROLE_QUESTION)
    node_hops.append(0)

    # Enforce hard cap.
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]

    # Labels: -100 everywhere, supervise only the answer token (and optionally
    # the two text bodies when mask_target_text is on).
    labels = [-100] * len(input_ids)
    if mask_target_text:
        # u_text body
        u_body_start = u_text_start + len(paper_a_prefix)
        u_body_end = min(u_body_start + len(paper_a_body), len(labels))
        for p in range(u_body_start, u_body_end):
            labels[p] = input_ids[p]
        # v_text body
        v_body_start = v_text_start + len(paper_b_prefix)
        v_body_end = min(v_body_start + len(paper_b_body), len(labels))
        for p in range(v_body_start, v_body_end):
            labels[p] = input_ids[p]

    if label_token_pos < len(labels):
        labels[label_token_pos] = answer_token

    return {
        "input_ids": input_ids,
        "labels": labels,
        "node_spans": node_spans,
        "node_hops": node_hops,
        "node_roles": node_roles,
        "cls_label": cls_label,
        "label_token_pos": label_token_pos,
        "answer_len": 1,
    }


def _truncate_text_for_neighbor(node_data: dict, nb_id: int) -> str:
    nd = node_data.get(nb_id)
    if nd is None:
        return ""
    return f"{nd['title']}. {nd['abstract']}"


def _sample_lp_neighbors(
    adj: dict[int, list[int]],
    u: int,
    v: int,
    max_neighbors_per_hop: int,
    max_hops: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Sample k-hop neighbours for u and v. Drops v from u's pool and u from
    v's pool (direct candidate-edge leak protection). Shared neighbours are
    deliberately NOT deduplicated: positive pairs share more neighbours than
    negative pairs, and removing them from one side would shrink that side's
    visible neighbour count in a label-correlated way, letting the model cheat
    by counting tokens instead of reading content."""
    u_ids, u_hops = _sample_khop_neighbors(adj, u, max_neighbors_per_hop, max_hops, rng)
    u_kept = [(nb, hop) for nb, hop in zip(u_ids, u_hops) if nb != v]
    v_ids, v_hops = _sample_khop_neighbors(adj, v, max_neighbors_per_hop, max_hops, rng)
    v_kept = [(nb, hop) for nb, hop in zip(v_ids, v_hops) if nb != u]

    u_ids2 = [nb for nb, _ in u_kept]
    u_hops2 = [hop for _, hop in u_kept]
    v_ids2 = [nb for nb, _ in v_kept]
    v_hops2 = [hop for _, hop in v_kept]
    return u_ids2, u_hops2, v_ids2, v_hops2


def _build_lp_samples(
    pos_edges: list[tuple[int, int]],
    neg_edges: list[tuple[int, int]],
    node_data: dict[int, dict],
    adj_train: dict[int, list[int]],
    tokenizer,
    max_seq_len: int,
    max_neighbors_per_hop: int,
    max_hops: int,
    seed: int,
    mask_target_text: bool,
    max_samples: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    samples: list[dict] = []

    pairs = [((u, v), 1) for u, v in pos_edges] + [((u, v), 0) for u, v in neg_edges]
    rng.shuffle(pairs)

    for (u, v), label in pairs:
        if max_samples and len(samples) >= max_samples:
            break
        if u not in node_data or v not in node_data:
            continue
        u_nb_ids, u_nb_hops, v_nb_ids, v_nb_hops = _sample_lp_neighbors(
            adj_train, u, v, max_neighbors_per_hop, max_hops, rng
        )
        u_nb_texts = [_truncate_text_for_neighbor(node_data, nb) for nb in u_nb_ids]
        v_nb_texts = [_truncate_text_for_neighbor(node_data, nb) for nb in v_nb_ids]
        u_text = f"{node_data[u]['title']}. {node_data[u]['abstract']}"
        v_text = f"{node_data[v]['title']}. {node_data[v]['abstract']}"

        sample = build_edge_sample(
            u_text=u_text,
            v_text=v_text,
            u_neighbor_texts=u_nb_texts,
            u_neighbor_hops=u_nb_hops,
            v_neighbor_texts=v_nb_texts,
            v_neighbor_hops=v_nb_hops,
            cls_label=label,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_target_text=mask_target_text,
        )
        sample["edge"] = (int(u), int(v))
        samples.append(sample)
    return samples


def load_lp_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    max_seq_len: int = 2048,
    max_neighbors_per_hop: int = MAX_NEIGHBORS_PER_HOP,
    max_hops: int = MAX_HOPS,
    seed: int = 42,
    neg_ratio: int = 1,
    mask_target_text: bool = False,
    max_samples: int = 0,
) -> Dataset:
    """Load a link-prediction dataset and return an HF Dataset of LP samples.

    Supports ``cora`` and ``ogbn-arxiv``.
    """
    if dataset_name == "cora":
        from dllm.data.datasets import cora_lp as lp_loader
    elif dataset_name == "ogbn-arxiv":
        from dllm.data.datasets import arxiv_lp as lp_loader
    else:
        raise NotImplementedError(
            f"LP loader supports 'cora' and 'ogbn-arxiv'; got {dataset_name!r}"
        )

    config = DATASET_CONFIGS[dataset_name]
    bundle = lp_loader.load(
        config, split=split, seed=seed, neg_ratio=neg_ratio
    )

    samples = _build_lp_samples(
        pos_edges=bundle["pos_edges"],
        neg_edges=bundle["neg_edges"],
        node_data=bundle["node_data"],
        adj_train=bundle["adj_train"],
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_neighbors_per_hop=max_neighbors_per_hop,
        max_hops=max_hops,
        seed=seed,
        mask_target_text=mask_target_text,
        max_samples=max_samples,
    )
    for s in samples:
        s["dataset"] = dataset_name

    dataset = Dataset.from_list(samples)
    dataset.info.description = (
        f"LP {dataset_name} ({split}) "
        f"pos={len(bundle['pos_edges'])} neg={len(bundle['neg_edges'])}"
    )
    return dataset
