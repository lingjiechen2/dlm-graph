"""Phase 7 — N=5000 ensemble analysis on Phase 6 caches.

Loads phase6_*.npz and runs the same ensemble methods (E1/E4/E5) plus
post-cal sweep, but on the 16 N=5000 caches (12 unique after dedup since
ckpt-2042 == ckpt-final).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


CACHE = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/logits_cache")
WORK = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128")
RESULTS_MD = WORK / "RESULTS.md"


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def load_phase6():
    caches = {}
    gt = None
    log_prior = None
    for f in sorted(CACHE.glob("phase6_*.npz")):
        n = f.stem.replace("phase6_", "")
        d = np.load(f)
        caches[n] = d["scores"]
        if gt is None:
            gt = d["cls_labels"]; log_prior = d["log_prior"]
        else:
            assert np.array_equal(d["cls_labels"], gt), f"cls_labels mismatch in {f.name}"
    return caches, gt, log_prior


def acc(pred, gt):
    return float(100 * (pred == gt).mean())


def vote_e5(arr, N, K):
    vote = np.zeros((N, K))
    for s in arr:
        sm = softmax(s, -1); conf = sm.max(axis=1); argm = s.argmax(-1)
        for i in range(N): vote[i, argm[i]] += conf[i]
    return vote


def vote_e4(arr, N):
    preds = np.stack([s.argmax(-1) for s in arr], axis=0)
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        u, c = np.unique(preds[:, i], return_counts=True)
        out[i] = u[c.argmax()]
    return out


def main():
    caches, gt, log_prior = load_phase6()
    print(f"loaded {len(caches)} caches; gt N={len(gt)}")
    N = len(gt); K = caches[next(iter(caches))].shape[1]

    # Standalone
    print(f"\n{'cache':<20} {'raw':>7}")
    for n, s in caches.items():
        print(f"  {n:<18} {acc(s.argmax(-1), gt):>7.2f}")

    from phase1c_full_test_prior import full_test_log_prior_shift
    new_shift, _, _ = full_test_log_prior_shift(log_prior, gt, K=K)

    # ckpt-final == ckpt-2042 (md5-confirmed); excluded from subsets.
    subsets = {
        "best_single (1845_nb10)": ["1845_nb10"],
        "1845_4nb": ["1845_nb10", "1845_nb12", "1845_nb15", "1845_nb30"],
        "3ckpts_nb10": ["1640_nb10", "1845_nb10", "2042_nb10"],
        "3ckpts_nb12": ["1640_nb12", "1845_nb12", "2042_nb12"],
        "3ckpts_x_4nb_12": [
            "1640_nb10", "1845_nb10", "2042_nb10",
            "1640_nb12", "1845_nb12", "2042_nb12",
            "1640_nb15", "1845_nb15", "2042_nb15",
            "1640_nb30", "1845_nb30", "2042_nb30",
        ],
        "1845_2042_nb10_nb15": ["1845_nb10", "1845_nb15", "2042_nb10", "2042_nb15"],
    }

    print(f"\n{'subset':<28} {'n':>3} {'mean':>7} {'E4':>7} {'E5':>7} {'E5+0.0':>7} {'E5+0.1':>7} {'E5+0.2':>7} {'E5+0.5':>7}")
    rows = []
    for name, members in subsets.items():
        arr = [caches[m] for m in members if m in caches]
        if not arr: continue
        mean_pool = acc(np.mean(arr, axis=0).argmax(-1), gt)
        e4 = acc(vote_e4(arr, N), gt)
        vote = vote_e5(arr, N, K)
        e5_raw = acc(vote.argmax(-1), gt)
        accs_cal = []
        for tau in [0.0, 0.1, 0.2, 0.5]:
            pred = (vote - tau * new_shift).argmax(-1)
            accs_cal.append(acc(pred, gt))
        print(f"{name:<28} {len(arr):>3} {mean_pool:>7.2f} {e4:>7.2f} {e5_raw:>7.2f} " + " ".join(f'{a:>7.2f}' for a in accs_cal))
        rows.append({"subset": name, "n": len(arr), "mean": mean_pool, "E4": e4, "E5": e5_raw,
                     "E5+0.0": accs_cal[0], "E5+0.1": accs_cal[1], "E5+0.2": accs_cal[2], "E5+0.5": accs_cal[3]})

    md = ["", f"## Phase 7 — N=5000 ensemble (auto " + os.popen("date '+%F %T'").read().strip() + ")",
          "",
          "Phase 6 produced 12 independent N=5000 caches (4 ckpts × 4 nb; ckpt-final == ckpt-2042 confirmed).",
          "σ at N=5000 ≈ 0.6 pt (vs 1.4 at N=1000). Best single: ckpt-1845 nb=10 = 75.56.",
          "",
          "| subset | n | mean | E4 | E5 | E5+τ=0.0 | E5+τ=0.1 | E5+τ=0.2 | E5+τ=0.5 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['subset']} | {r['n']} | {r['mean']:.2f} | {r['E4']:.2f} | {r['E5']:.2f} | {r['E5+0.0']:.2f} | {r['E5+0.1']:.2f} | {r['E5+0.2']:.2f} | {r['E5+0.5']:.2f} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nappended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
