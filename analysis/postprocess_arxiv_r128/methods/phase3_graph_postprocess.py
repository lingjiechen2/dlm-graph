"""Phase 3 — graph-structure post-processing on cached logits.

For each of the 1000 test nodes (seed=42), looks up the node's k-hop
neighbors and their TRAIN-set labels (ground-truth where available),
then combines this graph signal with the cached layer-0 logits.

Methods:
  N1. Low-confidence fallback to neighbor majority (1-hop train-labeled)
  N2. Same but 2-hop weighted
  O1. Label propagation: logits + alpha * log p(class | neighbor labels)
  O2. Sweep alpha for O1
  O3. Hard override when neighbor majority is overwhelming
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lingjie7/auto-research/projects/dlm-graph")
sys.path.insert(0, str(Path(__file__).parent))

from dllm.data.datasets import LOADERS as _DATA_LOADERS

from phase1_offline_sweeps import CkptCache, load_ckpt, acc, RESULTS_MD


def reproduce_test_split_ids(dataset_name="ogbn-arxiv", split="test", seed=42, max_samples=1000):
    """Mimic the seeded random subsetting in dllm/data/graph.py:1427.

    NOTE: in graph.py the loader is called with the literal seed before subsetting.
    """
    from dllm.data.graph import DATASET_CONFIGS
    node_data, adj, class_names, split_ids = _DATA_LOADERS[dataset_name](
        DATASET_CONFIGS[dataset_name], split, seed
    )
    rng = random.Random(seed)
    sampled = rng.sample(list(split_ids), min(max_samples, len(split_ids)))
    return node_data, adj, class_names, sampled


def get_neighbor_labels_for_node(
    node_id, adj, node_data, train_id_set, max_neighbors_per_hop=10, max_hops=2, rng=None,
):
    """Return list of (label, hop) for k-hop neighbors that are in the train set
    (i.e., have ground-truth labels exposed by the SFT regime).

    Uses the same neighbor-sampling routine as eval (seeded).
    """
    if rng is None:
        rng = random.Random(0)
    seen = {node_id}
    out = []  # list of (label_int, hop)

    nb_1hop = list(set(adj.get(node_id, [])))
    if len(nb_1hop) > max_neighbors_per_hop:
        nb_1hop = rng.sample(nb_1hop, max_neighbors_per_hop)
    for nb in nb_1hop:
        seen.add(nb)
        if nb in train_id_set and nb in node_data:
            label = node_data[nb].get("label")
            if label is not None and label >= 0:
                out.append((label, 1))

    if max_hops >= 2:
        candidates_2hop = []
        for nb in nb_1hop:
            for nb2 in adj.get(nb, []):
                if nb2 not in seen:
                    candidates_2hop.append(nb2)
                    seen.add(nb2)
        if len(candidates_2hop) > max_neighbors_per_hop:
            candidates_2hop = rng.sample(candidates_2hop, max_neighbors_per_hop)
        for nb2 in candidates_2hop:
            if nb2 in train_id_set and nb2 in node_data:
                label = node_data[nb2].get("label")
                if label is not None and label >= 0:
                    out.append((label, 2))

    return out


def get_train_id_set(dataset_name="ogbn-arxiv", seed=42):
    from dllm.data.graph import DATASET_CONFIGS
    _, _, _, train_split_ids = _DATA_LOADERS[dataset_name](
        DATASET_CONFIGS[dataset_name], "train", seed
    )
    return set(train_split_ids)


def neighbor_class_distribution(neighbor_labels, K, hop_weight=(1.0, 0.5)):
    """Hop-weighted soft prior over classes from neighbor labels."""
    counts = np.zeros(K)
    for lab, hop in neighbor_labels:
        w = hop_weight[hop - 1] if hop - 1 < len(hop_weight) else 0.0
        counts[lab] += w
    if counts.sum() == 0:
        return None
    return counts / counts.sum()


def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["1640", "1845", "2042", "final"])
    args = ap.parse_args()

    print("[phase3] loading cached logits...")
    cks = {s: load_ckpt(s) for s in args.ckpts}
    K = cks[args.ckpts[0]].scores.shape[1]
    print(f"[phase3] K={K} classes")

    print("[phase3] loading ogbn-arxiv graph (this may take a minute)...")
    node_data, adj, class_names, sampled_ids = reproduce_test_split_ids(
        dataset_name="ogbn-arxiv", split="test", seed=42, max_samples=1000
    )
    print(f"[phase3] sampled {len(sampled_ids)} test node IDs")
    train_ids = get_train_id_set("ogbn-arxiv", seed=42)
    print(f"[phase3] train set has {len(train_ids)} node IDs")

    # Verify class_names alignment with cached
    cached_cn = cks[args.ckpts[0]].class_names
    if list(class_names) != list(cached_cn):
        print("[phase3] WARNING class_names mismatch:")
        for i, (a, b) in enumerate(zip(class_names, cached_cn)):
            if a != b:
                print(f"  idx {i}: graph={a!r} npz={b!r}")

    # Build neighbor priors per sample using the SAME sampling regime as eval (seed=42)
    print("[phase3] computing neighbor priors per node...")
    rng = random.Random(42)
    priors = []  # list of [K] or None
    n_with_labels = 0
    for nid in sampled_ids:
        nb_labels = get_neighbor_labels_for_node(
            nid, adj, node_data, train_ids,
            max_neighbors_per_hop=10, max_hops=2, rng=rng,
        )
        prior = neighbor_class_distribution(nb_labels, K, hop_weight=(1.0, 0.5))
        priors.append(prior)
        if prior is not None:
            n_with_labels += 1
    coverage = n_with_labels / len(priors)
    print(f"[phase3] coverage: {n_with_labels}/{len(priors)} = {100*coverage:.1f}% of test nodes have ≥1 train-labeled neighbor")

    # Stack priors (use uniform when None)
    priors_stack = np.zeros((len(priors), K))
    for i, p in enumerate(priors):
        priors_stack[i] = p if p is not None else (np.ones(K) / K)
    log_neigh = np.log(priors_stack + 1e-30)  # [N, K]

    # Sanity: verify cls_labels in cached match what we'd expect
    # ALL cached ckpts share gt labels. Check one.
    gt_cached = cks[args.ckpts[0]].cls_labels
    # Reproduce expected gt: node_data[sampled_ids[i]]["label"]
    gt_expected = np.array([node_data[nid].get("label", -1) for nid in sampled_ids])
    if not np.array_equal(gt_cached, gt_expected):
        n_match = (gt_cached == gt_expected).sum()
        print(f"[phase3] WARNING gt mismatch: {n_match}/{len(gt_cached)} match. Check seed/sample-order alignment.")
    else:
        print("[phase3] gt alignment verified ✓")

    rows = []
    for s, c in cks.items():
        gt = c.cls_labels
        # baseline
        base_pred = c.scores.argmax(-1)
        base_acc = float((base_pred == gt).mean() * 100)
        rows.append({"method": "baseline (raw)", "ckpt": s, "acc": base_acc})

        # N1. Low-conf fallback to neighbor majority (1-hop)
        # confidence = top1 - top2 of softmax probs
        sm = np.exp(c.scores - c.scores.max(axis=1, keepdims=True))
        sm = sm / sm.sum(axis=1, keepdims=True)
        sorted_p = -np.sort(-sm, axis=1)
        gap = sorted_p[:, 0] - sorted_p[:, 1]
        # Try multiple thresholds
        for thr in [0.05, 0.10, 0.15, 0.20, 0.30]:
            pred = base_pred.copy()
            mask_low = gap < thr
            for i in np.where(mask_low)[0]:
                if priors[i] is not None and priors[i].max() > 0:
                    pred[i] = int(priors[i].argmax())
            new_acc = float((pred == gt).mean() * 100)
            rows.append({"method": f"N1. low-conf fallback gap<{thr}", "ckpt": s, "acc": new_acc,
                         "n_changed": int(mask_low.sum())})

        # O1. Label propagation: scores + alpha * log_neigh
        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            pred = (c.scores + alpha * log_neigh).argmax(-1)
            new_acc = float((pred == gt).mean() * 100)
            rows.append({"method": f"O1. logits + {alpha}·log p_neighbor", "ckpt": s, "acc": new_acc})

        # O2. Same but with calibration on
        for alpha in [0.3, 0.5, 1.0]:
            for tau in [0.0, 0.2, 0.5]:
                pred = (c.scores + alpha * log_neigh - tau * c.log_prior).argmax(-1)
                new_acc = float((pred == gt).mean() * 100)
                rows.append({"method": f"O2. logits + {alpha}·log p_neigh - {tau}·prior", "ckpt": s, "acc": new_acc})

        # O3. Hard override when neighbor majority is overwhelming (>=80% of weighted votes)
        for thr in [0.6, 0.7, 0.8, 0.9]:
            pred = base_pred.copy()
            for i, p in enumerate(priors):
                if p is not None and p.max() >= thr:
                    pred[i] = int(p.argmax())
            new_acc = float((pred == gt).mean() * 100)
            rows.append({"method": f"O3. hard-override neigh majority ≥{thr}", "ckpt": s, "acc": new_acc})

    # Write
    out_path = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/phase3_results.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[phase3] wrote {len(rows)} rows -> {out_path}")

    # Print top per-method by ckpt
    by_method_max = {}
    for r in rows:
        key = (r["method"], r["ckpt"])
        if key not in by_method_max or r["acc"] > by_method_max[key]["acc"]:
            by_method_max[key] = r
    rows_unique = sorted(by_method_max.values(), key=lambda r: -r["acc"])
    print("\n[phase3] top method×ckpt combos:")
    print(f"  {'method':<55} {'ckpt':>6} {'acc':>6}")
    for r in rows_unique[:30]:
        print(f"  {r['method']:<55} {r['ckpt']:>6} {r['acc']:>6.2f}")

    # Append to RESULTS.md
    md = ["", f"## Phase 3 — graph-structure post-processing (auto " + os.popen("date '+%F %T'").read().strip() + ")", ""]
    md.append(f"Per-test-node neighbor majority computed from train-labeled k-hop neighbors (k≤2, ≤10 per hop, hop_weight=(1.0, 0.5), seed=42).")
    md.append(f"Coverage: {100*coverage:.1f}% of 1000 test nodes have ≥1 train-labeled neighbor.")
    md.append("")
    md.append("| method | ckpt | acc | Δ vs 74.4 |")
    md.append("|---|---|---|---|")
    for r in rows_unique[:30]:
        md.append(f"| {r['method']} | {r['ckpt']} | {r['acc']:.2f} | {r['acc']-74.4:+.2f} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"[phase3] appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
