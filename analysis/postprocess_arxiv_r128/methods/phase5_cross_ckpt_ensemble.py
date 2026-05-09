"""Phase 5 — cross-ckpt × cross-setting TTA ensemble.

Pools forward-passes across multiple ckpts AND settings:
  - 4 ckpts (1640, 1845, 2042, final) × baseline default
  - 3 ckpts × {nb=12, nb=15, nb=30} from Phase 2c
  - + 1845's full Phase 2 + 2b + 2d caches

Reports E1 (mean-pool), E4 (plurality vote), E5 (confidence-weighted vote)
for various subset selections.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase1_offline_sweeps import load_ckpt, acc, RESULTS_MD


CACHE = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/logits_cache")


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def load_all_caches() -> dict[str, np.ndarray]:
    """Load every npz under logits_cache/ into a dict keyed by readable name."""
    out = {}
    # Phase 0 (per-ckpt baseline default settings)
    for f in sorted(CACHE.glob("ckpt-*.npz")):
        name = f"base_{f.stem.replace('ckpt-', '')}"
        out[name] = np.load(f)["scores"]
    # Phase 2 (1845 setting variants)
    for f in sorted(CACHE.glob("phase2_*_ckpt1845.npz")):
        n = f.stem.replace("phase2_", "").replace("_ckpt1845", "")
        out[f"1845_{n}"] = np.load(f)["scores"]
    # Phase 2b (1845)
    for f in sorted(CACHE.glob("phase2b_*_ckpt1845.npz")):
        n = f.stem.replace("phase2b_", "").replace("_ckpt1845", "")
        out[f"1845_{n}"] = np.load(f)["scores"]
    # Phase 2c (other ckpts × nb)
    for f in sorted(CACHE.glob("phase2c_*.npz")):
        n = f.stem.replace("phase2c_", "")
        out[n] = np.load(f)["scores"]
    # Phase 2d (neighbor jitter, 1845)
    for f in sorted(CACHE.glob("phase2d_*_ckpt1845.npz")):
        n = f.stem.replace("phase2d_", "").replace("_ckpt1845", "")
        out[f"1845_{n}"] = np.load(f)["scores"]
    return out


def vote_e5(caches_list, gt, N, K):
    vote = np.zeros((N, K))
    for s in caches_list:
        sm = softmax(s, -1); conf = sm.max(axis=1); argm = s.argmax(-1)
        for i in range(N): vote[i, argm[i]] += conf[i]
    return acc(vote.argmax(-1), gt)


def vote_e4(caches_list, gt, N, K):
    preds = np.stack([s.argmax(-1) for s in caches_list], axis=0)
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        u, c = np.unique(preds[:, i], return_counts=True)
        out[i] = u[c.argmax()]
    return acc(out, gt)


def mean_pool(caches_list, gt):
    s = np.mean(caches_list, axis=0)
    return acc(s.argmax(-1), gt)


def main():
    base_1845 = load_ckpt("1845")
    gt = base_1845.cls_labels
    N, K = base_1845.scores.shape

    all_caches = load_all_caches()
    print(f"Loaded {len(all_caches)} setting × ckpt caches")
    for k in sorted(all_caches.keys())[:30]:
        a = acc(all_caches[k].argmax(-1), gt)
        print(f"  {k:<35} {a:.2f}")
    if len(all_caches) > 30:
        print(f"  ... ({len(all_caches)-30} more)")

    # Define subsets to evaluate
    available = set(all_caches.keys())
    def avail(names):
        return [all_caches[n] for n in names if n in available]

    subsets = {
        "all-available": list(all_caches.keys()),
        "best1845": ["base_1845", "1845_s2_nb15", "1845_nb12", "1845_nb30"],
        "cross_ckpt_baseline": ["base_1640", "base_1845", "base_2042", "base_final"],
        "cross_ckpt_nb12": ["ckpt1640_nb12", "ckpt2042_nb12", "ckptfinal_nb12", "1845_nb12"],
        "cross_ckpt_nb15": ["ckpt1640_nb15", "ckpt2042_nb15", "ckptfinal_nb15", "1845_s2_nb15"],
        "cross_ckpt_nb30": ["ckpt1640_nb30", "ckpt2042_nb30", "ckptfinal_nb30", "1845_nb30"],
        "16-cross_ckpt_x_nb": [
            "base_1640", "base_1845", "base_2042", "base_final",
            "ckpt1640_nb12", "ckpt2042_nb12", "ckptfinal_nb12", "1845_nb12",
            "ckpt1640_nb15", "ckpt2042_nb15", "ckptfinal_nb15", "1845_s2_nb15",
            "ckpt1640_nb30", "ckpt2042_nb30", "ckptfinal_nb30", "1845_nb30",
        ],
        "12-3ckpts_x_4nb_no_baseline": [
            "ckpt1640_nb12", "ckpt2042_nb12", "ckptfinal_nb12",
            "ckpt1640_nb15", "ckpt2042_nb15", "ckptfinal_nb15",
            "ckpt1640_nb30", "ckpt2042_nb30", "ckptfinal_nb30",
            "1845_nb12", "1845_s2_nb15", "1845_nb30",
        ],
        # Phase 2d jitters
        "1845_nb10_4jit_only": ["1845_nb10_jit7", "1845_nb10_jit13", "1845_nb10_jit23", "1845_nb10_jit31"],
        "1845_nb15_4jit_only": ["1845_nb15_jit7", "1845_nb15_jit13", "1845_nb15_jit23", "1845_nb15_jit31"],
        "1845_nb10_jit_plus_base": [
            "base_1845",
            "1845_nb10_jit7", "1845_nb10_jit13", "1845_nb10_jit23", "1845_nb10_jit31",
        ],
        "1845_nb10_8jit_combined": [
            "base_1845",  # = nb=10 with seed=42 (original)
            "1845_nb10_jit7", "1845_nb10_jit13", "1845_nb10_jit23", "1845_nb10_jit31",
            "1845_nb15_jit7", "1845_nb15_jit13", "1845_nb15_jit23", "1845_nb15_jit31",
        ],
        "best1845_plus_jitters": [
            "base_1845", "1845_s2_nb15", "1845_nb12", "1845_nb30",
            "1845_nb10_jit7", "1845_nb10_jit13", "1845_nb10_jit23", "1845_nb10_jit31",
            "1845_nb15_jit7", "1845_nb15_jit13", "1845_nb15_jit23", "1845_nb15_jit31",
        ],
        "ALL_24_passes": [
            "base_1640", "base_1845", "base_2042", "base_final",
            "ckpt1640_nb12", "ckpt2042_nb12", "ckptfinal_nb12", "1845_nb12",
            "ckpt1640_nb15", "ckpt2042_nb15", "ckptfinal_nb15", "1845_s2_nb15",
            "ckpt1640_nb30", "ckpt2042_nb30", "ckptfinal_nb30", "1845_nb30",
            "1845_nb10_jit7", "1845_nb10_jit13", "1845_nb10_jit23", "1845_nb10_jit31",
            "1845_nb15_jit7", "1845_nb15_jit13", "1845_nb15_jit23", "1845_nb15_jit31",
        ],
    }

    print(f"\n{'subset':<35} {'n':>3} {'mean':>6} {'E4':>6} {'E5':>6} {'Δ-E5':>6}")
    rows = []
    for name, members in subsets.items():
        arr = avail(members)
        n_avail = len(arr)
        n_total = len(members)
        if n_avail == 0:
            continue
        mp = mean_pool(arr, gt)
        e4 = vote_e4(arr, gt, N, K)
        e5 = vote_e5(arr, gt, N, K)
        rows.append({"subset": name, "n_avail": n_avail, "n_total": n_total,
                     "mean": mp, "E4": e4, "E5": e5})
        print(f"{name:<35} {n_avail:>3}/{n_total:<2} {mp:>6.2f} {e4:>6.2f} {e5:>6.2f} {e5-74.4:>+6.2f}")

    # Append to RESULTS.md
    md = ["", f"## Phase 5 — cross-ckpt + cross-setting ensemble (auto " + os.popen("date '+%F %T'").read().strip() + ")", ""]
    md.append(f"Pools forward passes across ckpts (1640/1845/2042/final) and settings (default + nb={12,15,30}).")
    md.append("")
    md.append("| subset | n | mean-pool | E4 plur | E5 conf-vote | Δ-E5 vs 74.4 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(f"| {r['subset']} | {r['n_avail']}/{r['n_total']} | {r['mean']:.2f} | {r['E4']:.2f} | **{r['E5']:.2f}** | {r['E5']-74.4:+.2f} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[phase5] appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
