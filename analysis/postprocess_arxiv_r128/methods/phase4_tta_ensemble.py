"""Phase 4 — test-time-augmentation ensemble across Phase 2 settings.

Combines per-sample logits from baseline (ckpt-1845, default settings) and the
Phase 2 setting variants (s1..s7) to form augmented predictions.

Methods:
  E1. Mean-pool across all available settings (raw logits)
  E2. Mean-pool then global tau-cal sweep
  E3. CV-fit per-setting weights
  E4. Top-K-confidence vote (each setting predicts; majority wins)
  E5. Confidence-weighted vote (weight each setting's prediction by its softmax max)

Compares to per-setting standalone baseline (each setting eval'd alone).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from phase1_offline_sweeps import CkptCache, load_ckpt, acc, RESULTS_MD


WORK = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128")
CACHE = WORK / "logits_cache"


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def load_phase2_caches(include_phase2b=True, exclude_substrings=("nbfirst",)) -> dict[str, CkptCache]:
    """Load all Phase 2 / 2b setting npz files for ckpt-1845.

    Excludes settings whose name contains any of the exclude_substrings (default
    drops nbfirst, which is catastrophic at 47%).
    """
    out = {}
    patterns = [str(CACHE / "phase2_*_ckpt1845.npz")]
    if include_phase2b:
        patterns.append(str(CACHE / "phase2b_*_ckpt1845.npz"))
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            base = os.path.basename(f)
            name = base.replace("phase2_", "").replace("phase2b_", "").replace("_ckpt1845.npz", "")
            if any(x in name for x in exclude_substrings):
                continue
            d = np.load(f, allow_pickle=True)
            out[name] = CkptCache(
                step=name,
                scores=d["scores"],
                cls_labels=d["cls_labels"],
                log_prior=d["log_prior"] if "log_prior" in d.files else np.zeros(d["scores"].shape[1]),
                per_pos=d["per_pos_logprobs"] if "per_pos_logprobs" in d.files else None,
                class_names=[str(x) for x in d["class_names"].tolist()],
                pred_raw=d["predictions_raw"],
                pred_cal=d["predictions_cal"],
            )
    return out


def main():
    # Baseline: ckpt-1845 with default settings
    base = load_ckpt("1845")
    p2 = load_phase2_caches()
    print(f"[phase4] baseline: 1845 (raw acc={acc(base.scores.argmax(-1), base.cls_labels):.2f})")
    print(f"[phase4] phase2 settings loaded: {list(p2.keys())}")
    if not p2:
        print("[phase4] no phase 2 caches — aborting")
        return

    # Verify all share same cls_labels
    for name, c in p2.items():
        if not np.array_equal(c.cls_labels, base.cls_labels):
            n_match = (c.cls_labels == base.cls_labels).sum()
            print(f"[phase4] WARN: {name} cls_labels mismatch ({n_match}/{len(base.cls_labels)})")

    gt = base.cls_labels

    # Standalone per-setting acc
    rows = []
    rows.append({"method": "Standalone baseline (1845, defaults)", "ckpt": "1845", "acc": acc(base.scores.argmax(-1), gt)})
    for name, c in p2.items():
        rows.append({"method": f"Standalone {name}", "ckpt": "1845", "acc": acc(c.scores.argmax(-1), gt)})

    # E1. Mean-pool all settings (incl baseline) raw
    all_caches = [base] + list(p2.values())
    all_names = ["baseline"] + list(p2.keys())
    all_scores = np.stack([c.scores for c in all_caches], axis=0)  # [M, N, K]
    mean_raw = all_scores.mean(axis=0)
    rows.append({"method": f"E1. mean-pool all ({len(all_caches)} settings, raw)", "ckpt": "1845",
                 "acc": acc(mean_raw.argmax(-1), gt)})
    rows.append({"method": f"E1. mean-pool all (cal tau=1)", "ckpt": "1845",
                 "acc": acc((mean_raw - base.log_prior).argmax(-1), gt)})

    # E2. Mean-pool + tau sweep
    best_e2 = (-1, None)
    for tau in np.linspace(-0.5, 1.5, 21):
        pred = (mean_raw - tau * base.log_prior).argmax(-1)
        a = acc(pred, gt)
        if a > best_e2[0]:
            best_e2 = (a, tau)
    rows.append({"method": f"E2. mean-pool + best tau (={best_e2[1]:.2f}, oracle)", "ckpt": "1845",
                 "acc": best_e2[0]})

    # E3. CV-fit weights
    rng = np.random.RandomState(7)
    M, N, K = all_scores.shape
    idx = rng.permutation(N)
    fold_acc = []
    weights_log = []
    for fold in range(5):
        te = idx[fold * N // 5:(fold + 1) * N // 5]
        tr = np.setdiff1d(idx, te)
        w = np.zeros(M)
        for _ in range(300):
            wn = np.exp(w - w.max()); wn /= wn.sum()
            mixed = (wn[:, None, None] * all_scores[:, tr, :]).sum(0)
            p = softmax(mixed, -1)
            gh = np.zeros_like(p); gh[np.arange(len(tr)), gt[tr]] = 1.0
            dwn = np.array([((p - gh) * all_scores[m, tr, :]).sum(1).mean() for m in range(M)])
            dw = wn * (dwn - (wn * dwn).sum())
            w -= 0.1 * dw
        wn = np.exp(w - w.max()); wn /= wn.sum()
        weights_log.append(wn)
        mixed_te = (wn[:, None, None] * all_scores[:, te, :]).sum(0)
        fold_acc.append(acc(mixed_te.argmax(-1), gt[te]))
    avg_w = np.mean(weights_log, axis=0)
    label = ", ".join(f"{n}={w:.2f}" for n, w in zip(all_names, avg_w))
    rows.append({"method": f"E3. CV-fit weights ({label})", "ckpt": "1845",
                 "acc": float(np.mean(fold_acc))})

    # E4. Plurality vote across settings (each setting raw-argmax)
    preds_per = np.stack([c.scores.argmax(-1) for c in all_caches], axis=0)  # [M, N]
    vote_pred = np.zeros(N, dtype=np.int64)
    for i in range(N):
        # most common (ties → lowest class id)
        unique, counts = np.unique(preds_per[:, i], return_counts=True)
        vote_pred[i] = unique[counts.argmax()]
    rows.append({"method": "E4. plurality vote (raw)", "ckpt": "1845", "acc": acc(vote_pred, gt)})

    # E5. Confidence-weighted vote: each setting contributes argmax with weight = softmax max prob
    confidence_logits = np.zeros((N, K))
    for c in all_caches:
        s = softmax(c.scores, -1)
        conf = s.max(axis=1)  # [N]
        argm = c.scores.argmax(-1)  # [N]
        for i in range(N):
            confidence_logits[i, argm[i]] += conf[i]
    rows.append({"method": "E5. confidence-weighted vote", "ckpt": "1845",
                 "acc": acc(confidence_logits.argmax(-1), gt)})

    # E6. Use cal-mean instead of raw-mean
    mean_cal_each = []
    for c in all_caches:
        mean_cal_each.append(c.scores - c.log_prior)
    mean_cal = np.stack(mean_cal_each, axis=0).mean(axis=0)
    rows.append({"method": "E6. mean-pool then argmax-of-cal-each (mean-after-cal)", "ckpt": "1845",
                 "acc": acc(mean_cal.argmax(-1), gt)})

    rows.sort(key=lambda r: -r["acc"])
    print(f"\n[phase4] sorted results:")
    print(f"  {'method':<70} {'acc':>6}")
    for r in rows:
        print(f"  {r['method']:<70} {r['acc']:>6.2f}")

    out_path = WORK / "phase4_results.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md = ["", f"## Phase 4 — test-time augmentation ensemble (auto " + os.popen("date '+%F %T'").read().strip() + ")",
          "",
          f"Combines baseline ckpt-1845 with Phase 2 setting variants ({len(p2)} settings).",
          f"All evaluated on the same 1000 test samples (seed=42).",
          "",
          "| method | acc | Δ vs 74.4 |",
          "|---|---|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['acc']:.2f} | {r['acc']-74.4:+.2f} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"[phase4] appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
