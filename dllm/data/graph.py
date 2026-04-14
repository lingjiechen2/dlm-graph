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
    "ogbn-arxiv": {
        "ogb_root": HF_CACHE_ROOT / "ogb___ogbn-arxiv" / "ogbn_arxiv",
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


# ---------------------------------------------------------------------------
# Prompt / sample construction
# ---------------------------------------------------------------------------


def get_answer_labels(num_classes: int) -> list[str]:
    """
    Generate answer labels for MC format.

    Always uses numeric labels: "0", "1", ..., "N-1".
    For ≤10 classes these are single tokens; for >10 they are multi-token
    but still natural MC answers that LLMs handle well.
    """
    return [str(i) for i in range(num_classes)]


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
        nb_prefix = _tok(f"\nNeighbor {i + 1}: ")
        nb_body = _tok(nb_text)
        nb_ids = (nb_prefix + nb_body)[:per_nb_budget]
        nb_token_list.append((nb_ids, hop))

    # --- Assemble content tokens (inside user turn) based on layout ---
    if prompt_layout == "neighbor_first":
        content_toks = []
        nb_content_spans = []  # spans relative to content_toks start
        nb_hops = []
        max_nb_total = content_budget - target_content_len

        for nb_ids, hop in nb_token_list:
            if len(content_toks) + len(nb_ids) > max_nb_total:
                break
            start = len(content_toks)
            content_toks.extend(nb_ids)
            nb_content_spans.append([start, len(content_toks)])
            nb_hops.append(hop)

        target_content_start = len(content_toks)
        content_toks.extend(target_content_toks)
        target_content_span = [target_content_start, len(content_toks)]
    else:
        # target_first (default)
        content_toks = list(target_content_toks)
        target_content_span = [0, target_content_len]
        nb_content_spans = []
        nb_hops = []

        for nb_ids, hop in nb_token_list:
            if len(content_toks) >= content_budget:
                break
            start = len(content_toks)
            content_toks.extend(nb_ids)
            nb_content_spans.append([start, len(content_toks)])
            nb_hops.append(hop)

    # --- Now wrap content in chat template ---
    # Decode content tokens back to text, then apply chat template
    content_text = tokenizer.decode(content_toks, skip_special_tokens=False)
    template_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content_text}],
        tokenize=True,
        add_generation_prompt=True,
    )

    # Find where content tokens start within the template
    # Template structure: [BOS, start_header, "user", end_header, \n, \n, ...content..., eot, start_header, "assistant", end_header, \n, \n]
    # We need to locate the content tokens within template_ids
    # The user content starts after the first \n\n (positions 4,5 in the template)
    # and ends before <|eot_id|>

    # Find content offset by matching: after "user" header there are two \n tokens
    # Template prefix: BOS + start_header + "user" + end_header + \n + \n
    empty_prefix_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # Find where content would be inserted (between prefix \n\n and eot_id)
    # In empty template, the eot_id comes right after the \n\n of user header
    eot_id = tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0]
    # Content offset = index of eot_id in empty template (content goes before it)
    content_offset = empty_prefix_ids.index(eot_id)

    # Append answer tokens after the template (assistant header end)
    input_ids = list(template_ids) + answer_tokens
    label_token_pos = len(template_ids)  # answer starts right after template

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
) -> dict:
    """
    Build a sample using natural category infill format.

    Format:
        Paper: <target_text>
        Options: 0) Case Based 1) Genetic Algorithms ...
        The category of this paper is: <class_name>

    The class name tokens are the answer (masked during eval).
    All class names are padded to max_answer_tokens with pad_token_id.
    """

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # --- Build answer tokens (class name, padded to max_answer_tokens) ---
    class_name = class_names[cls_label]
    answer_tokens = _tok(class_name)[:max_answer_tokens]
    # Pad to max_answer_tokens
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    answer_len = len(answer_tokens)
    while len(answer_tokens) < max_answer_tokens:
        answer_tokens.append(pad_id)

    # --- Tokenize target section ---
    target_prefix = _tok("Paper: ")
    target_body = _tok(target_node_text)
    # Build options string (same as mc_digit but without digit prefix)
    options_str = " ".join(f"{i}) {name}" for i, name in enumerate(class_names))
    options_suffix = _tok(f"\nOptions: {options_str}")
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
        nb_prefix = _tok(f"\nNeighbor {i + 1}: ")
        nb_body = _tok(nb_text)
        nb_ids = (nb_prefix + nb_body)[:per_nb_budget]
        nb_token_list.append((nb_ids, hop))

    # --- Assemble based on layout ---
    # For category_infill, answer is always at the end of target section.
    # target_first: [target+answer] [nb1] [nb2] ...
    # neighbor_first: [nb1] [nb2] ... [target+answer]
    if prompt_layout == "neighbor_first":
        input_ids = []
        nb_spans = []
        nb_hops = []
        max_nb_total = max_seq_len - target_section_len

        for nb_ids, hop in nb_token_list:
            if len(input_ids) + len(nb_ids) > max_nb_total:
                break
            start = len(input_ids)
            input_ids.extend(nb_ids)
            nb_spans.append([start, len(input_ids)])
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

        for nb_ids, hop in nb_token_list:
            if len(input_ids) >= max_seq_len:
                break
            start = len(input_ids)
            input_ids.extend(nb_ids)
            node_spans.append([start, len(input_ids)])
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
    # Answer tokens in labels (only real tokens, not padding)
    for j in range(answer_len):
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
        "answer_len": answer_len,
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

    ``prompt_layout`` controls ordering (target_first vs neighbor_first).

    Labels are -100 everywhere EXCEPT at the answer token positions.
    """

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
        )

    # --- mc_digit format (original) ---

    def _tok(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # --- Build options string ---
    if answer_labels is None:
        answer_labels = get_answer_labels(len(class_names))
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
        )

    # --- Tokenize target section ---
    target_prefix = _tok("Paper: ")
    target_body = _tok(target_node_text)
    options_prefix = _tok(f"\nOptions: {options_str}\nAnswer: ")

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
        nb_prefix = _tok(f"\nNeighbor {i + 1}: ")
        nb_body = _tok(nb_text)
        nb_ids = (nb_prefix + nb_body)[:per_nb_budget]
        nb_token_list.append((nb_ids, hop))

    # --- Assemble based on layout ---
    if prompt_layout == "neighbor_first":
        input_ids = []
        nb_spans = []
        nb_hops = []
        max_nb_total = max_seq_len - target_section_len

        for nb_ids, hop in nb_token_list:
            if len(input_ids) + len(nb_ids) > max_nb_total:
                break
            start = len(input_ids)
            input_ids.extend(nb_ids)
            nb_spans.append([start, len(input_ids)])
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

        for nb_ids, hop in nb_token_list:
            if len(input_ids) >= max_seq_len:
                break
            start = len(input_ids)
            input_ids.extend(nb_ids)
            node_spans.append([start, len(input_ids)])
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
) -> list[dict]:
    """Build TM-DLM samples for a list of node IDs (shared across all datasets)."""
    answer_labels = get_answer_labels(len(class_names))
    rng = random.Random(seed)
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
            answer_labels=answer_labels,
            max_answer_tokens=max_answer_tokens,
            prompt_layout=prompt_layout,
            use_chat_template=use_chat_template,
            prompt_format=prompt_format,
        )
        samples.append(result)

    return samples


# ---------------------------------------------------------------------------
# Dataset-specific data loaders (each returns node_data, adj, class_names, split_ids)
# ---------------------------------------------------------------------------


def _load_cora_data(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    """Load Cora data from HuggingFace TAPE + PyG Planetoid."""
    split_map = {"train": "train", "val": "validation", "test": "test"}
    hf_split = split_map[split]

    tape_ds = load_dataset(
        config["hf_path"],
        cache_dir=str(config["cache_dir"]),
    )

    # Build full text lookup
    node_data: dict[int, dict] = {}
    label_to_class: dict[int, str] = {}
    for s in tape_ds:
        for sample in tape_ds[s]:
            node_data[sample["id"]] = {
                "title": sample["T"],
                "abstract": sample["A"],
                "label": sample["label"],
            }
            label_to_class[sample["label"]] = sample["class"]

    class_names = [label_to_class[i] for i in range(len(label_to_class))]

    # Graph structure
    from torch_geometric.datasets import Planetoid

    pyg_ds = Planetoid(root=str(config["pyg_root"]), name=config["pyg_name"])
    adj = _build_adjacency(pyg_ds[0].edge_index.numpy())

    split_ids = [sample["id"] for sample in tape_ds[hf_split]]

    return node_data, adj, class_names, split_ids


def _load_pubmed_data(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    """Load PubMed data from local TAPE files (tab + JSON + citations)."""
    import json as _json

    tape_dir = Path(config["tape_dir"])
    class_names = config["class_names"]

    # 1. Parse node file: position -> (pmid, label)
    node_file = tape_dir / "data" / "Pubmed-Diabetes.NODE.paper.tab"
    with open(node_file) as f:
        lines = f.readlines()

    node_data: dict[int, dict] = {}
    all_ids = []
    for i, line in enumerate(lines[2:]):
        parts = line.strip().split("\t")
        pmid = int(parts[0])
        label = int(parts[1].split("=")[1]) - 1  # 1-3 -> 0-2
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


def _load_ogbn_arxiv_data(
    config: dict,
    split: str,
    seed: int,
) -> tuple[dict, dict, list[str], list[int]]:
    """Load ogbn-arxiv data from OGB files + titleabs.tsv."""
    import gzip
    import csv

    ogb_root = Path(config["ogb_root"])

    # 1. Load node index -> paper ID mapping
    mapping_file = ogb_root / "mapping" / "nodeidx2paperid.csv.gz"
    nodeidx_to_paperid = {}
    with gzip.open(mapping_file, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            nodeidx_to_paperid[int(row[0])] = int(row[1])

    # 2. Load paper ID -> (title, abstract) from titleabs.tsv
    titleabs_file = ogb_root / "raw" / "titleabs.tsv"
    paperid_to_text = {}
    with open(titleabs_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                paperid_to_text[int(parts[0])] = {
                    "title": parts[1],
                    "abstract": parts[2],
                }
            elif len(parts) == 2:
                paperid_to_text[int(parts[0])] = {
                    "title": parts[1],
                    "abstract": "",
                }

    # 3. Load node labels
    label_file = ogb_root / "raw" / "node-label.csv.gz"
    node_labels = {}
    with gzip.open(label_file, "rt") as f:
        for idx, line in enumerate(f):
            node_labels[idx] = int(line.strip())

    # 4. Class names
    class_names = config["class_names"]

    # 5. Load edges
    edge_file = ogb_root / "raw" / "edge.csv.gz"
    src_list, dst_list = [], []
    with gzip.open(edge_file, "rt") as f:
        for line in f:
            parts = line.strip().split(",")
            s, d = int(parts[0]), int(parts[1])
            src_list.extend([s, d])
            dst_list.extend([d, s])
    edge_index = np.array([src_list, dst_list])
    adj = _build_adjacency(edge_index)

    # 6. Load split
    split_name = {"train": "train", "val": "valid", "test": "test"}[split]
    split_file = ogb_root / "split" / "time" / f"{split_name}.csv.gz"
    split_ids = []
    with gzip.open(split_file, "rt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("node"):
                split_ids.append(int(line))

    # 7. Build node_data lookup
    node_data = {}
    for node_idx, paper_id in nodeidx_to_paperid.items():
        text_info = paperid_to_text.get(paper_id, {"title": "", "abstract": ""})
        node_data[node_idx] = {
            "title": text_info["title"],
            "abstract": text_info["abstract"],
            "label": node_labels.get(node_idx, 0),
        }

    return node_data, adj, class_names, split_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DATA_LOADERS = {
    "cora": _load_cora_data,
    "pubmed": _load_pubmed_data,
    "ogbn-arxiv": _load_ogbn_arxiv_data,
}


def load_tag_dataset(
    dataset_name: str,
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
) -> Dataset:
    """
    Load a TAG dataset and return a HuggingFace Dataset of TM-DLM samples.

    Supports: "cora", "pubmed", "ogbn-arxiv".
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: {list(DATASET_CONFIGS.keys())}"
        )

    node_data, adj, class_names, split_ids = _DATA_LOADERS[dataset_name](
        DATASET_CONFIGS[dataset_name], split, seed
    )

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
    )

    dataset = Dataset.from_list(samples)
    dataset.info.description = (
        f"TM-DLM {dataset_name} ({split}), {len(class_names)} classes"
    )
    return dataset


def get_class_token_ids(
    dataset_name: str,
    tokenizer,
    max_answer_tokens: int = 1,
    prompt_format: str = "mc_digit",
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

    if "class_names" in config:
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
    answer_labels = get_answer_labels(num_classes)

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
