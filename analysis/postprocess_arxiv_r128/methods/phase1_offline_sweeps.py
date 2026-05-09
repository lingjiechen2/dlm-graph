"""Phase 1 offline post-processing sweeps on cached logits.

Reads `analysis/postprocess_arxiv_r128/logits_cache/ckpt-{step}.npz`
files and runs CPU-only logit transforms. Results appended to RESULTS.md.

Methods:
  A: calibration on/off (free re-read)
  B: temperature sweep on calibration shift, tau in {0, 0.3, 0.5, 0.7, 1, 1.5, 2}
  C: logits ensemble across ckpts (mean / top-2 / weighted)
  D: per-answer-position aggregation (mean / sum / first / second / max)
  F: softmax-then-log re-normalize before calibration
  H: top-k restricted argmax (drop tail-class logits to -inf before argmax)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WORKDIR = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128")
CACHE_DIR = WORKDIR / "logits_cache"
RESULTS_MD = WORKDIR / "RESULTS.md"


@dataclass
class CkptCache:
    step: str
    scores: np.ndarray  # [N, K] mean log-prob across answer positions
    cls_labels: np.ndarray  # [N]
    log_prior: np.ndarray  # [K] = log p_train - log p_test
    per_pos: np.ndarray | None  # [N, max_ans, K] or None
    class_names: list[str]
    pred_raw: np.ndarray
    pred_cal: np.ndarray


def load_ckpt(step: str) -> CkptCache:
    p = CACHE_DIR / f"ckpt-{step}.npz"
    d = np.load(p, allow_pickle=True)
    return CkptCache(
        step=step,
        scores=d["scores"],
        cls_labels=d["cls_labels"],
        log_prior=d["log_prior"] if "log_prior" in d.files else np.zeros(d["scores"].shape[1]),
        per_pos=d["per_pos_logprobs"] if "per_pos_logprobs" in d.files else None,
        class_names=[str(x) for x in d["class_names"].tolist()],
        pred_raw=d["predictions_raw"],
        pred_cal=d["predictions_cal"],
    )


def acc(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(100.0 * (pred == gt).sum() / len(gt))


def method_A_calibration(cks: list[CkptCache]) -> list[dict]:
    out = []
    for c in cks:
        out.append({
            "method": "A. baseline (mean-pos, raw)",
            "ckpt": c.step,
            "acc": acc(c.scores.argmax(-1), c.cls_labels),
        })
        out.append({
            "method": "A. baseline (mean-pos, calibrated)",
            "ckpt": c.step,
            "acc": acc((c.scores - c.log_prior).argmax(-1), c.cls_labels),
        })
    return out


def method_B_temperature(cks: list[CkptCache]) -> list[dict]:
    out = []
    taus = [-0.5, -0.3, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    for c in cks:
        for tau in taus:
            pred = (c.scores - tau * c.log_prior).argmax(-1)
            out.append({
                "method": f"B. temp tau={tau}",
                "ckpt": c.step,
                "acc": acc(pred, c.cls_labels),
            })
    return out


def method_B2_uniform_target(cks: list[CkptCache]) -> list[dict]:
    """Calibrate against UNIFORM target (only correct train prior, not test shift).

    Need train log-prob from log_prior + log p_test. log_prior = log p_train - log p_test
    so log p_train = log_prior + log p_test. We don't have log p_test directly.
    But we approximate: use empirical class freq in this 1000 to recover log p_test.
    """
    out = []
    for c in cks:
        K = c.scores.shape[1]
        # Recover log p_test from the test-sample class label distribution
        counts = np.bincount(c.cls_labels, minlength=K).astype(np.float64)
        eps = 1.0
        p_test = (counts + eps) / (counts.sum() + eps * K)
        log_p_test = np.log(p_test)
        log_p_train = c.log_prior + log_p_test
        # Subtract log_p_train (uniform target = subtract only train prior)
        for tau in [0.3, 0.5, 0.7, 1.0]:
            pred = (c.scores - tau * log_p_train).argmax(-1)
            out.append({"method": f"B2. uniform-target tau={tau}", "ckpt": c.step,
                        "acc": acc(pred, c.cls_labels)})
    return out


def method_D2_pos_weight_sweep(cks: list[CkptCache]) -> list[dict]:
    """Sweep weighting between answer positions: alpha * pos0 + (1-alpha) * pos1."""
    out = []
    for c in cks:
        if c.per_pos is None or c.per_pos.shape[1] != 2:
            continue
        pp = c.per_pos
        gt = c.cls_labels
        for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            s = alpha * pp[:, 0, :] + (1 - alpha) * pp[:, 1, :]
            out.append({"method": f"D2. pos-weight alpha={alpha:.1f} (raw)", "ckpt": c.step,
                        "acc": acc(s.argmax(-1), gt)})
    return out


def method_I_self_cal(cks: list[CkptCache]) -> list[dict]:
    """Self-calibration: subtract softmax-mean of model's own scores across the 1000 samples
    (corrects for model's over-confident classes without needing external prior)."""
    out = []
    for c in cks:
        s = c.scores
        # softmax over classes per sample → average across samples → log
        s_max = s.max(axis=1, keepdims=True)
        sm = np.exp(s - s_max)
        sm = sm / sm.sum(axis=1, keepdims=True)
        avg_p = sm.mean(axis=0)  # [K]
        log_avg = np.log(avg_p + 1e-30)
        for tau in [0.3, 0.5, 0.7, 1.0]:
            pred = (s - tau * log_avg).argmax(-1)
            out.append({"method": f"I. self-cal tau={tau}", "ckpt": c.step,
                        "acc": acc(pred, c.cls_labels)})
    return out


def method_J_cv_tau(cks: list[CkptCache]) -> list[dict]:
    """5-fold CV: find optimal tau per ckpt on 4/5 of data, eval on held-out 1/5."""
    out = []
    rng = np.random.RandomState(0)
    for c in cks:
        N = len(c.cls_labels)
        idx = rng.permutation(N)
        fold_acc = []
        best_taus = []
        for fold in range(5):
            te = idx[fold * N // 5 : (fold + 1) * N // 5]
            tr = np.setdiff1d(idx, te)
            taus = np.linspace(-0.5, 1.5, 41)
            best_tau, best_a = 0.0, -1.0
            for tau in taus:
                pred_tr = (c.scores[tr] - tau * c.log_prior).argmax(-1)
                a = (pred_tr == c.cls_labels[tr]).mean()
                if a > best_a:
                    best_a, best_tau = a, tau
            best_taus.append(best_tau)
            pred_te = (c.scores[te] - best_tau * c.log_prior).argmax(-1)
            fold_acc.append((pred_te == c.cls_labels[te]).mean())
        out.append({"method": f"J. CV-tau (mean tau={np.mean(best_taus):.2f})",
                    "ckpt": c.step,
                    "acc": float(100.0 * np.mean(fold_acc))})
    return out


def method_C_ensemble(cks: list[CkptCache]) -> list[dict]:
    out = []
    if len(cks) < 2:
        return out
    # Verify all share the same cls_labels (same seed → same samples)
    for c in cks[1:]:
        assert np.array_equal(c.cls_labels, cks[0].cls_labels), f"cls_labels mismatch on {c.step}"
    gt = cks[0].cls_labels

    # Mean of all ckpts (raw)
    mean_scores = np.mean([c.scores for c in cks], axis=0)
    out.append({"method": f"C. ensemble-mean (n={len(cks)}, raw)", "ckpt": "ALL",
                "acc": acc(mean_scores.argmax(-1), gt)})
    # Mean of all ckpts (cal)
    log_prior = cks[0].log_prior  # all same
    out.append({"method": f"C. ensemble-mean (n={len(cks)}, cal)", "ckpt": "ALL",
                "acc": acc((mean_scores - log_prior).argmax(-1), gt)})

    # Top-2 ensemble per sample (mean of top-2-acc-ckpts)
    accs = [(c.step, acc(c.scores.argmax(-1), gt), c) for c in cks]
    accs.sort(key=lambda x: -x[1])
    top2 = [accs[0][2], accs[1][2]]
    mean2 = np.mean([t.scores for t in top2], axis=0)
    out.append({"method": f"C. ensemble-top2 ({top2[0].step}+{top2[1].step}, raw)", "ckpt": "TOP2",
                "acc": acc(mean2.argmax(-1), gt)})
    out.append({"method": f"C. ensemble-top2 ({top2[0].step}+{top2[1].step}, cal)", "ckpt": "TOP2",
                "acc": acc((mean2 - log_prior).argmax(-1), gt)})

    # Top-3 ensemble
    if len(accs) >= 3:
        top3 = [accs[i][2] for i in range(3)]
        mean3 = np.mean([t.scores for t in top3], axis=0)
        out.append({"method": f"C. ensemble-top3 ({'+'.join(t.step for t in top3)}, cal)",
                    "ckpt": "TOP3",
                    "acc": acc((mean3 - log_prior).argmax(-1), gt)})

    # Weighted by per-ckpt baseline acc (softmax over accs as weights)
    raw_accs = np.array([a for _, a, _ in accs])
    w = np.exp((raw_accs - raw_accs.max()) / 1.0)
    w = w / w.sum()
    weighted = np.zeros_like(cks[0].scores)
    for (step, _, c), wi in zip(accs, w):
        weighted += wi * c.scores
    out.append({"method": f"C. ensemble-weighted-by-acc (cal)", "ckpt": "ALL",
                "acc": acc((weighted - log_prior).argmax(-1), gt)})
    return out


def method_D_pos_agg(cks: list[CkptCache]) -> list[dict]:
    out = []
    for c in cks:
        if c.per_pos is None:
            continue
        pp = c.per_pos  # [N, T, K]
        N, T, K = pp.shape
        gt = c.cls_labels
        log_prior = c.log_prior
        # mean (baseline)
        s_mean = pp.mean(axis=1)
        # sum
        s_sum = pp.sum(axis=1)
        # first-only
        s_first = pp[:, 0, :]
        # last-only (typically position 1 = the second digit)
        s_last = pp[:, -1, :]
        # max-position
        s_max = pp.max(axis=1)
        # min-position
        s_min = pp.min(axis=1)
        for name, s in [("mean", s_mean), ("sum", s_sum), ("first", s_first),
                        ("last", s_last), ("max", s_max), ("min", s_min)]:
            out.append({"method": f"D. pos-agg={name} (cal)", "ckpt": c.step,
                        "acc": acc((s - log_prior).argmax(-1), gt)})
    return out


def method_F_renorm_then_cal(cks: list[CkptCache]) -> list[dict]:
    out = []
    for c in cks:
        # softmax over classes, then log, then subtract prior
        s = c.scores
        s_max = s.max(axis=1, keepdims=True)
        sm = np.exp(s - s_max)
        sm = sm / sm.sum(axis=1, keepdims=True)
        log_p = np.log(sm + 1e-30)
        pred = (log_p - c.log_prior).argmax(-1)
        out.append({"method": "F. softmax-renorm + cal", "ckpt": c.step,
                    "acc": acc(pred, c.cls_labels)})
    return out


def method_H_topk(cks: list[CkptCache]) -> list[dict]:
    out = []
    for c in cks:
        for k in [3, 5, 10, 20]:
            s = c.scores - c.log_prior  # cal first
            topk_idx = np.argpartition(-s, kth=k, axis=1)[:, :k]
            mask = np.full_like(s, -np.inf)
            np.put_along_axis(mask, topk_idx, np.take_along_axis(s, topk_idx, 1), axis=1)
            pred = mask.argmax(-1)
            out.append({"method": f"H. top-k={k} (after cal)", "ckpt": c.step,
                        "acc": acc(pred, c.cls_labels)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["1640", "1845", "2042", "final"])
    ap.add_argument("--out", default=str(WORKDIR / "phase1_results.jsonl"))
    args = ap.parse_args()

    print(f"[phase1] loading ckpts from {CACHE_DIR}")
    cks = []
    for s in args.ckpts:
        try:
            c = load_ckpt(s)
            cks.append(c)
            print(f"  ckpt-{s}: scores={c.scores.shape} per_pos={None if c.per_pos is None else c.per_pos.shape} N={len(c.cls_labels)}")
        except FileNotFoundError as e:
            print(f"  ckpt-{s}: MISSING ({e})")
    if not cks:
        raise SystemExit("no ckpt cache found")

    all_results = []
    all_results += method_A_calibration(cks)
    all_results += method_B_temperature(cks)
    all_results += method_B2_uniform_target(cks)
    all_results += method_C_ensemble(cks)
    all_results += method_D_pos_agg(cks)
    all_results += method_D2_pos_weight_sweep(cks)
    all_results += method_F_renorm_then_cal(cks)
    all_results += method_H_topk(cks)
    all_results += method_I_self_cal(cks)
    all_results += method_J_cv_tau(cks)

    # Write JSONL
    with open(args.out, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"[phase1] wrote {len(all_results)} rows -> {args.out}")

    # Print top-of-leaderboard
    by_method_max = {}
    for r in all_results:
        m = r["method"]
        if m not in by_method_max or r["acc"] > by_method_max[m]["acc"]:
            by_method_max[m] = r
    rows = sorted(by_method_max.values(), key=lambda r: -r["acc"])
    print("\n[phase1] top method×ckpt combos:")
    print(f"  {'method':<55} {'ckpt':>6} {'acc':>6}")
    for r in rows[:25]:
        print(f"  {r['method']:<55} {r['ckpt']:>6} {r['acc']:>6.2f}")

    # Append a markdown table to RESULTS.md
    md = ["", "## Phase 1 — auto-generated " + os.popen("date '+%F %T'").read().strip(), ""]
    md.append("Top-25 method×ckpt combos by accuracy (1000-sample, seed=42).")
    md.append("Baseline-raw = 74.4% (ckpt-1845).")
    md.append("")
    md.append(f"| method | ckpt | acc | Δ vs 74.4 |")
    md.append(f"|---|---|---|---|")
    for r in rows[:25]:
        d = r["acc"] - 74.4
        md.append(f"| {r['method']} | {r['ckpt']} | {r['acc']:.2f} | {d:+.2f} |")
    md.append("")
    md.append(f"Full per-row log: `{args.out}` ({len(all_results)} rows)")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"[phase1] appended top-25 to {RESULTS_MD}")


if __name__ == "__main__":
    main()
