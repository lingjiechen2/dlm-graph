"""How many neighbors actually fit into seq=4096 for arxiv?

Replays the real SFT pipeline (sample_khop + build_node_sample) on a
random subsample of the arxiv train split and records, per node:
  - num_sampled_nbs:     |neighbors returned by sampler| (≤ 1 + 10 + 10*10 = 110)
  - num_nbs_kept:        |neighbors whose tokens actually made it into input_ids|
  - num_nbs_truncated:   neighbors kept but body got cut (per_nb_budget < full text)
  - target_body_chars:   tokens of target abstract that fit (vs raw)
  - per_nb_budget:       max(remaining // N, 20)

Run:
  /home/lingjie7/anaconda3/envs/dllm/bin/python analysis/seq_budget_arxiv.py
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/lingjie7/auto-research/projects/dlm-graph")
sys.path.insert(0, str(REPO))

from dllm.data.graph import _sample_khop_neighbors, build_node_sample, _truncate_text_for_neighbor

SEQ_LEN = 4096
N_SAMPLES = 500
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

print("[load] arxiv processed_data.pt …")
data = torch.load(REPO / ".datasets/llaga/ogbn-arxiv/processed_data.pt", weights_only=False)
y = data.y.tolist()
train_mask = data.train_mask.tolist()
title = data.title
abs_ = data.abs
label_texts = list(data.label_texts)

# Build node_data and adjacency dict in the format graph.py expects
edge_index = data.edge_index.numpy()
adj: dict[int, list[int]] = {}
for s, d in edge_index.T:
    adj.setdefault(int(s), []).append(int(d))
    adj.setdefault(int(d), []).append(int(s))
for k in adj:
    adj[k] = list(set(adj[k]))

node_data: dict[int, dict] = {}
for i in range(len(y)):
    node_data[i] = {"title": title[i], "abstract": abs_[i], "label": int(y[i])}

train_ids = [i for i, m in enumerate(train_mask) if m]
sample_ids = random.sample(train_ids, k=min(N_SAMPLES, len(train_ids)))

# Tokenizer
print("[load] tokenizer …")
from dllm.utils import get_tokenizer
class _MA:
    model_name_or_path = "GSAI-ML/LLaDA-8B-Instruct"
tok = get_tokenizer(model_args=_MA())

# Stats accumulators
n_sampled_nbs: list[int] = []
n_kept_nbs: list[int] = []
n_truncated_nbs: list[int] = []
per_nb_budgets: list[int] = []
target_body_kept: list[int] = []
target_body_full: list[int] = []
total_used: list[int] = []
abs_lens: list[int] = []

for nid in sample_ids:
    # 1-hop and 2-hop sample, cap=10 each, matching SFT settings
    rng = random.Random(SEED + nid)
    nb_ids, nb_hops = _sample_khop_neighbors(
        adj=adj, node_id=nid, max_neighbors_per_hop=10, max_hops=2, rng=rng,
    )
    nb_texts = []
    for nb_id in nb_ids:
        nb_texts.append(_truncate_text_for_neighbor(node_data, nb_id))
    target_text = node_data[nid].get("title", "") + ". " + node_data[nid].get("abstract", "")

    # Build sample with the real pipeline
    sample = build_node_sample(
        target_node_text=target_text,
        neighbor_texts=nb_texts,
        neighbor_hops=nb_hops,
        cls_label=node_data[nid]["label"],
        class_names=label_texts,
        tokenizer=tok,
        max_seq_len=SEQ_LEN,
        prompt_format="mc_digit",
        answer_label_style="digit0",
        max_answer_tokens=2,
        prompt_layout="target_first",
        include_neighbor_labels=False,
    )

    input_ids = sample["input_ids"]
    node_spans = sample["node_spans"]
    node_hops = sample["node_hops"]

    # Span 0 = target. Remaining = neighbors that survived.
    nb_spans = node_spans[1:]
    n_sampled_nbs.append(len(nb_ids))
    n_kept_nbs.append(len(nb_spans))

    # Truncated check: count neighbors whose span length < full nb token length
    truncated = 0
    for i, (s, e) in enumerate(nb_spans):
        if i >= len(nb_texts): break
        full_tok = len(tok.encode(nb_texts[i], add_special_tokens=False))
        # nb prefix is small (< 5 tokens for "Neighbor i: "). Approx span vs full_tok+~3
        if (e - s) < full_tok + 2:
            truncated += 1
    n_truncated_nbs.append(truncated)

    # per_nb_budget reconstruction
    target_section_len = node_spans[0][1] - node_spans[0][0]
    remaining = SEQ_LEN - target_section_len
    n = max(len(nb_ids), 1)
    per_nb = max(remaining // n, 20)
    per_nb_budgets.append(per_nb)
    total_used.append(len(input_ids))

    # Target abstract truncation
    target_body_full.append(len(tok.encode(target_text, add_special_tokens=False)))
    target_body_kept.append(target_section_len)  # this includes prefix+options+answer
    abs_lens.append(len(tok.encode(node_data[nid]["abstract"], add_special_tokens=False)))


def stats(arr, name):
    a = np.array(arr)
    print(f"  {name:30}  mean={a.mean():.1f}  median={np.median(a):.1f}  p10={np.percentile(a,10):.1f}  p90={np.percentile(a,90):.1f}  max={a.max():.0f}")

print(f"\n=== arxiv seq={SEQ_LEN}, N={len(sample_ids)} train nodes, max_nb_per_hop=10, max_hops=2 ===")
stats(n_sampled_nbs, "neighbors sampled (out of 20 cap: 10+10)")
stats(n_kept_nbs,    "neighbors kept in input_ids")
stats(n_truncated_nbs, "neighbors with body TRUNCATED")
stats(per_nb_budgets, "per_nb_budget tokens")
stats(target_body_full, "target abstract+title tokens (raw)")
stats(target_body_kept, "target section tokens (kept incl. prefix)")
stats(abs_lens, "abstract tokens (raw, no title)")
stats(total_used, "total input_ids tokens used")

# Distribution buckets for nb retention
kept = np.array(n_kept_nbs); samp = np.array(n_sampled_nbs)
ratio = np.where(samp > 0, kept / samp, 1.0)
print(f"\n  retention ratio (kept / sampled):")
print(f"    mean={ratio.mean()*100:.1f}%   median={np.median(ratio)*100:.1f}%")
print(f"    samples retaining 100%: {(ratio>=0.999).sum()}/{len(ratio)} ({(ratio>=0.999).mean()*100:.1f}%)")
print(f"    samples retaining <50%: {(ratio<0.5).sum()}/{len(ratio)} ({(ratio<0.5).mean()*100:.1f}%)")

trunc = np.array(n_truncated_nbs); kep = np.array(n_kept_nbs)
trunc_share = np.where(kep > 0, trunc / kep, 0)
print(f"\n  per-sample share of KEPT neighbors that were truncated:")
print(f"    mean={trunc_share.mean()*100:.1f}%   median={np.median(trunc_share)*100:.1f}%")
print(f"    samples with 0 truncated kept-neighbors: {(trunc==0).sum()}/{len(trunc)} ({(trunc==0).mean()*100:.1f}%)")
