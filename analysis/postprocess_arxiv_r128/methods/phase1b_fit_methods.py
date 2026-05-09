"""Phase 1b — fit-based offline methods on cached logits.

Builds on Phase 1 (sweeps) with cross-validated parameter fitting:
  K: per-class bias (Platt-style, gradient-descent fit on 4/5 then eval on 1/5)
  L: per-class temperature (multiplicative scaling per class) via CV
  M: optimal mix (alpha) of raw vs cal logits via 1-D scan + CV
  N: ensemble fit — learn per-ckpt weights on 4/5, eval on 1/5
  P: per-class accuracy diagnostic (which classes are bottlenecks)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Reuse Phase 1 loader
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phase1_offline_sweeps import CkptCache, load_ckpt, acc, RESULTS_MD


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def neg_loglik(b, scores, gt):
    s = scores - b  # b is [K]
    p = softmax(s, axis=-1)
    return -np.log(p[np.arange(len(gt)), gt] + 1e-30).mean()


def fit_per_class_bias(scores, gt, n_iter=400, lr=0.05, l2=0.01):
    """Gradient descent on per-class bias b ∈ R^K minimizing CE(softmax(s-b), gt).

    Derivation: s' = s - b, p = softmax(s'). Standard CE+softmax: dL/d s'_k = p_k - 1[k=gt].
    Chain rule: d s'/d b = -I → dL/d b_k = -(p_k - 1[k=gt]) = 1[k=gt] - p_k = (gh - p)_k.

    Adding L2: total dL/db = (gh - p).mean(axis=0) + l2 * b.
    GD update: b -= lr * dL/db.

    Concrete check: K=2, gt=0, scores=[0,0], b=[0,0], p=[0.5,0.5], gh=[1,0].
    grad = (gh - p) = [0.5, -0.5]. b -= lr * grad → b = [-lr*0.5, +lr*0.5].
    Then s' = [s_0 + lr*0.5, s_1 - lr*0.5] → p_0 grows. ✓
    """
    K = scores.shape[1]
    b = np.zeros(K)
    for _ in range(n_iter):
        s = scores - b
        p = softmax(s, axis=-1)
        gh = np.zeros_like(p)
        gh[np.arange(len(gt)), gt] = 1.0
        grad = (gh - p).mean(axis=0) + l2 * b
        b = b - lr * grad
    return b


def method_K_per_class_bias(cks: list[CkptCache]) -> list[dict]:
    out = []
    rng = np.random.RandomState(0)
    for c in cks:
        N = len(c.cls_labels)
        idx = rng.permutation(N)
        fold_acc = []
        for fold in range(5):
            te = idx[fold * N // 5 : (fold + 1) * N // 5]
            tr = np.setdiff1d(idx, te)
            b = fit_per_class_bias(c.scores[tr], c.cls_labels[tr])
            pred = (c.scores[te] - b).argmax(-1)
            fold_acc.append((pred == c.cls_labels[te]).mean())
        out.append({"method": "K. per-class-bias (5-fold CV fit)",
                    "ckpt": c.step,
                    "acc": float(100.0 * np.mean(fold_acc))})
    return out


def fit_alpha_mix(scores, log_prior, gt):
    """1-D scan over alpha in argmax(scores - alpha * log_prior). Returns best alpha by accuracy."""
    alphas = np.linspace(-0.5, 1.5, 41)
    best_a, best_alpha = -1.0, 0.0
    for alpha in alphas:
        pred = (scores - alpha * log_prior).argmax(-1)
        a = (pred == gt).mean()
        if a > best_a:
            best_a, best_alpha = a, alpha
    return best_alpha, best_a


def method_M_alpha_cv(cks: list[CkptCache]) -> list[dict]:
    """5-fold CV: alpha on 4/5, eval on 1/5. Reports ensemble of folds."""
    out = []
    rng = np.random.RandomState(1)
    for c in cks:
        N = len(c.cls_labels)
        idx = rng.permutation(N)
        fold_acc = []
        for fold in range(5):
            te = idx[fold * N // 5 : (fold + 1) * N // 5]
            tr = np.setdiff1d(idx, te)
            best_alpha, _ = fit_alpha_mix(c.scores[tr], c.log_prior, c.cls_labels[tr])
            pred = (c.scores[te] - best_alpha * c.log_prior).argmax(-1)
            fold_acc.append((pred == c.cls_labels[te]).mean())
        out.append({"method": "M. alpha-mix CV (5-fold)",
                    "ckpt": c.step,
                    "acc": float(100.0 * np.mean(fold_acc))})
    return out


def method_N_ensemble_weighted_fit(cks: list[CkptCache]) -> list[dict]:
    """Fit per-ckpt weights (softmax of free params) via CE minimization on 4/5, eval on 1/5."""
    if len(cks) < 2:
        return []
    out = []
    rng = np.random.RandomState(2)
    gt = cks[0].cls_labels
    N = len(gt)
    K = cks[0].scores.shape[1]
    M = len(cks)
    # Stack scores: [M, N, K]
    S = np.stack([c.scores for c in cks], axis=0)
    log_prior = cks[0].log_prior
    idx = rng.permutation(N)
    fold_acc = []
    weights_log = []
    for fold in range(5):
        te = idx[fold * N // 5 : (fold + 1) * N // 5]
        tr = np.setdiff1d(idx, te)
        # Param: w (M-vector). softmax to get convex weights
        w = np.zeros(M)
        for _ in range(300):
            wn = np.exp(w - w.max()); wn = wn / wn.sum()
            mixed = (wn[:, None, None] * S[:, tr, :]).sum(axis=0)  # [|tr|, K]
            p = softmax(mixed, axis=-1)
            gh = np.zeros_like(p); gh[np.arange(len(tr)), gt[tr]] = 1.0
            # grad w.r.t. wn: dCE/dwn_m = mean over tr of <(p - gh), S_m>
            dwn = np.array([((p - gh) * S[m, tr, :]).sum(axis=1).mean() for m in range(M)])
            # softmax jacobian: dwn_m / dw_j = wn_m (delta_mj - wn_j)
            dw = wn * (dwn - (wn * dwn).sum())
            w -= 0.1 * dw
        wn = np.exp(w - w.max()); wn = wn / wn.sum()
        weights_log.append(wn.tolist())
        mixed_te = (wn[:, None, None] * S[:, te, :]).sum(axis=0)
        pred = mixed_te.argmax(-1)
        fold_acc.append((pred == gt[te]).mean())
    avg_w = np.mean(weights_log, axis=0)
    label = "+".join(f"{c.step}={w:.2f}" for c, w in zip(cks, avg_w))
    out.append({"method": f"N. weighted-ensemble fit (avg w: {label})",
                "ckpt": "ALL",
                "acc": float(100.0 * np.mean(fold_acc))})
    # Also try mix + cal
    fold_acc_cal = []
    for fold in range(5):
        te = idx[fold * N // 5 : (fold + 1) * N // 5]
        tr = np.setdiff1d(idx, te)
        # Use the already-found avg_w but per-fold optimal alpha
        mixed_tr = (avg_w[:, None, None] * S[:, tr, :]).sum(axis=0)
        best_alpha, _ = fit_alpha_mix(mixed_tr, log_prior, gt[tr])
        mixed_te = (avg_w[:, None, None] * S[:, te, :]).sum(axis=0)
        pred = (mixed_te - best_alpha * log_prior).argmax(-1)
        fold_acc_cal.append((pred == gt[te]).mean())
    out.append({"method": "N+M. weighted-ensemble + alpha-cal CV",
                "ckpt": "ALL",
                "acc": float(100.0 * np.mean(fold_acc_cal))})
    return out


def method_P_per_class_diag(cks: list[CkptCache]) -> list[dict]:
    """Diagnostic: per-class acc + top confusion target for each ckpt's baseline preds."""
    out = []
    for c in cks:
        if c.step != "1845":
            continue  # diagnostic on best ckpt only
        pred = c.scores.argmax(-1)
        K = c.scores.shape[1]
        per_class_n = np.bincount(c.cls_labels, minlength=K)
        per_class_acc = np.zeros(K)
        per_class_conf = np.zeros((K, K), dtype=int)
        for i in range(len(c.cls_labels)):
            t = c.cls_labels[i]; p = pred[i]
            per_class_conf[t, p] += 1
            if t == p: per_class_acc[t] += 1
        per_class_acc = per_class_acc / np.maximum(per_class_n, 1)
        # find top 5 classes by error count
        err = per_class_n - (per_class_acc * per_class_n).round().astype(int)
        order = np.argsort(-err)
        for cls in order[:8]:
            if per_class_n[cls] == 0: continue
            row = per_class_conf[cls].copy()
            row[cls] = 0
            top_pred = row.argmax()
            out.append({
                "method": f"P. diag {c.class_names[cls][:25]}",
                "ckpt": c.step,
                "acc": float(100.0 * per_class_acc[cls]),
                "n": int(per_class_n[cls]),
                "errors": int(err[cls]),
                "top_confusion_to": c.class_names[top_pred][:25],
                "top_confusion_n": int(row[top_pred]),
            })
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["1640", "1845", "2042", "final"])
    ap.add_argument("--out_jsonl", default=None)
    args = ap.parse_args()

    WORK = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128")
    out_path = args.out_jsonl or str(WORK / "phase1b_results.jsonl")

    cks = [load_ckpt(s) for s in args.ckpts]
    print(f"[phase1b] loaded {len(cks)} ckpts")

    rows = []
    rows += method_K_per_class_bias(cks)
    rows += method_M_alpha_cv(cks)
    rows += method_N_ensemble_weighted_fit(cks)
    rows += method_P_per_class_diag(cks)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[phase1b] wrote {len(rows)} rows -> {out_path}")

    # Print sorted (excluding diagnostic P)
    metric_rows = [r for r in rows if not r["method"].startswith("P.")]
    metric_rows.sort(key=lambda r: -r["acc"])
    print(f"\n[phase1b] fit methods (CV-evaluated):")
    print(f"  {'method':<55} {'ckpt':>6} {'acc':>6}")
    for r in metric_rows:
        print(f"  {r['method']:<55} {r['ckpt']:>6} {r['acc']:>6.2f}")

    print(f"\n[phase1b] per-class diagnostic (ckpt-1845 baseline):")
    print(f"  {'class':<28} {'n':>4} {'acc':>6} {'errors':>6} -> {'top confused with':<28} {'n':>4}")
    for r in rows:
        if r["method"].startswith("P."):
            cls = r["method"][8:]
            print(f"  {cls:<28} {r['n']:>4} {r['acc']:>6.2f} {r['errors']:>6} -> {r['top_confusion_to']:<28} {r['top_confusion_n']:>4}")

    md = ["", f"## Phase 1b — fit-based methods (auto " + __import__('os').popen("date '+%F %T'").read().strip() + ")", ""]
    md.append("All fit methods use 5-fold CV on the 1000 samples — train on 800, eval on 200, average.")
    md.append("")
    md.append("| method | ckpt | acc | Δ vs 74.4 |")
    md.append("|---|---|---|---|")
    for r in metric_rows:
        md.append(f"| {r['method']} | {r['ckpt']} | {r['acc']:.2f} | {r['acc']-74.4:+.2f} |")
    md.append("")
    md.append("### Per-class confusion (ckpt-1845, raw)")
    md.append("")
    md.append("| class | n | acc | errors | top confused with | n |")
    md.append("|---|---:|---:|---:|---|---:|")
    for r in rows:
        if r["method"].startswith("P."):
            cls = r["method"][8:]
            md.append(f"| {cls} | {r['n']} | {r['acc']:.2f} | {r['errors']} | {r['top_confusion_to']} | {r['top_confusion_n']} |")
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[phase1b] appended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
