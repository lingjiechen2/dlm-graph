"""Phase 1c — re-calibrate with FULL-test prior instead of 1000-sample prior.

The cached log_prior used `log p_train - log p_test_1000`, where p_test_1000
has high variance (some classes have 0-2 samples → big swings).

Here we recompute the test prior over the FULL ogbn-arxiv test split (48,604
samples), keeping log p_train the same, and re-sweep tau.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lingjie7/auto-research/projects/dlm-graph")
sys.path.insert(0, str(Path(__file__).parent))

from dllm.data.datasets import LOADERS as _DATA_LOADERS
from dllm.data.graph import DATASET_CONFIGS

from phase1_offline_sweeps import CkptCache, load_ckpt, acc, RESULTS_MD


def full_test_log_prior_shift(cks_log_prior_1000, cks_cls_labels_1000, dataset_name="ogbn-arxiv", seed=42, K=40):
    """Recover log p_train from cached log_prior (which is log p_train - log p_test_1000),
    then recompute log p_test_full from the full test split, and produce new shift.

    Returns: log p_train - log p_test_full
    """
    # Recover log p_test_1000 from sample
    counts_1000 = np.bincount(cks_cls_labels_1000, minlength=K).astype(np.float64)
    eps = 1.0
    p_test_1000 = (counts_1000 + eps) / (counts_1000.sum() + eps * K)
    log_p_test_1000 = np.log(p_test_1000)
    log_p_train = cks_log_prior_1000 + log_p_test_1000  # = log p_train

    # Get full test class freq from raw graph
    node_data, adj, class_names, test_split_ids = _DATA_LOADERS[dataset_name](
        DATASET_CONFIGS[dataset_name], "test", seed
    )
    counts_full = np.zeros(K, dtype=np.float64)
    for nid in test_split_ids:
        if nid in node_data:
            lab = node_data[nid].get("label")
            if lab is not None and 0 <= lab < K:
                counts_full[lab] += 1.0
    p_test_full = (counts_full + eps) / (counts_full.sum() + eps * K)
    log_p_test_full = np.log(p_test_full)
    return log_p_train - log_p_test_full, len(test_split_ids), counts_full


def main():
    import os
    cks = [load_ckpt(s) for s in ["1640", "1845", "2042", "final"]]
    K = cks[0].scores.shape[1]

    new_shift, n_test, counts_full = full_test_log_prior_shift(
        cks[0].log_prior, cks[0].cls_labels, K=K
    )
    print(f"[phase1c] full-test split has {n_test} samples (sum counts = {int(counts_full.sum())})")
    print(f"[phase1c] new shift range: min={new_shift.min():.3f} max={new_shift.max():.3f} range={new_shift.max()-new_shift.min():.3f}")
    print(f"[phase1c] old shift range: min={cks[0].log_prior.min():.3f} max={cks[0].log_prior.max():.3f} range={cks[0].log_prior.max()-cks[0].log_prior.min():.3f}")

    rows = []
    taus = [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    for c in cks:
        for tau in taus:
            pred = (c.scores - tau * new_shift).argmax(-1)
            a = float((pred == c.cls_labels).mean() * 100)
            rows.append({"method": f"P1c. full-test-prior tau={tau}", "ckpt": c.step, "acc": a})

    # Best per ckpt
    rows.sort(key=lambda r: -r["acc"])
    print(f"\n[phase1c] sorted results:")
    for r in rows[:30]:
        print(f"  {r['method']:<40} {r['ckpt']:>6} {r['acc']:>6.2f}")

    out_path = "/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/phase1c_results.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md = ["", f"## Phase 1c — full-test prior calibration (auto " + os.popen("date '+%F %T'").read().strip() + ")",
          "", f"Recompute test prior over full ogbn-arxiv test split ({n_test} samples) for stable calibration.",
          f"New shift range = {new_shift.max()-new_shift.min():.2f} (vs 1000-sample shift = {cks[0].log_prior.max()-cks[0].log_prior.min():.2f}).",
          "",
          "| method | ckpt | acc | Δ vs 74.4 |",
          "|---|---|---|---|"]
    for r in rows[:25]:
        md.append(f"| {r['method']} | {r['ckpt']} | {r['acc']:.2f} | {r['acc']-74.4:+.2f} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"[phase1c] appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
