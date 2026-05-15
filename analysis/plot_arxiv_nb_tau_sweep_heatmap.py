"""
Heatmap of arxiv ckpt-1845 (§21 r128) accuracy across a 2D sweep:
    x-axis  = max_neighbors_per_hop (8 values from Phase 2 + 2b)
    y-axis  = τ calibration coefficient applied to full-test-prior shift
    color   = accuracy at N=1000

Output: analysis/figures/arxiv_nb_tau_sweep_heatmap.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "postprocess_arxiv_r128" / "methods"))

from _plot_utils import apply_rcparams, load_style


CACHE = Path("/home/lingjie7/auto-research/projects/dlm-graph/analysis/postprocess_arxiv_r128/logits_cache")

# ckpt-1845 nb sweep caches (N=1000, mc_digit, nonb, default hops=2, topo=True)
NB_TO_CACHE = {
    5:  "phase2_s1_nb5_ckpt1845.npz",
    10: "ckpt-1845.npz",                     # Phase 0 baseline (default settings)
    12: "phase2b_nb12_ckpt1845.npz",
    15: "phase2_s2_nb15_ckpt1845.npz",
    18: "phase2b_nb18_ckpt1845.npz",
    20: "phase2_s3_nb20_ckpt1845.npz",
    25: "phase2b_nb25_ckpt1845.npz",
    30: "phase2b_nb30_ckpt1845.npz",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="analysis/plot_style.json")
    p.add_argument("--out", default="analysis/figures/arxiv_nb_tau_sweep_heatmap.png")
    args = p.parse_args()

    style = load_style(Path(args.style))
    apply_rcparams(style)

    from phase1c_full_test_prior import full_test_log_prior_shift

    # Load all caches, compute full-test prior shift (same across nb at fixed ckpt)
    base_npz = np.load(CACHE / NB_TO_CACHE[10])
    gt = base_npz["cls_labels"]
    K = base_npz["scores"].shape[1]
    log_prior_1k = base_npz["log_prior"]
    new_shift, _, _ = full_test_log_prior_shift(log_prior_1k, gt, K=K)

    nbs = sorted(NB_TO_CACHE.keys())
    taus = np.array([-0.3, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    acc_grid = np.zeros((len(taus), len(nbs)))

    for j, nb in enumerate(nbs):
        f = CACHE / NB_TO_CACHE[nb]
        d = np.load(f)
        s = d["scores"]
        gt_c = d["cls_labels"]
        assert np.array_equal(gt_c, gt), f"gt mismatch for nb={nb}"
        for i, tau in enumerate(taus):
            pred = (s - tau * new_shift).argmax(-1)
            acc_grid[i, j] = 100 * (pred == gt_c).mean()

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=style["figure"]["dpi"])

    # Diverging cmap centered at baseline (raw ckpt-1845 nb=10 acc)
    baseline_raw = acc_grid[np.where(taus == 0.0)[0][0], np.where(np.array(nbs) == 10)[0][0]]
    vmin, vmax = acc_grid.min(), acc_grid.max()
    half = max(baseline_raw - vmin, vmax - baseline_raw)
    norm = plt.Normalize(vmin=baseline_raw - half, vmax=baseline_raw + half)
    im = ax.imshow(
        acc_grid,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
    )

    ax.set_xticks(range(len(nbs)))
    ax.set_xticklabels([str(n) for n in nbs])
    ax.set_yticks(range(len(taus)))
    ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in taus])
    ax.set_xlabel(r"\texttt{max\_neighbors\_per\_hop}",
                  fontsize=style["axes"]["label_fontsize"])
    ax.set_ylabel(r"$\tau$ (calibration coefficient)",
                  fontsize=style["axes"]["label_fontsize"])
    ax.set_title(
        r"ogbn-arxiv ckpt-1845, nb $\times$ $\tau$ sweep (N=1000, $\sigma\approx1.4$ pt)",
        fontsize=style["title"]["fontsize"],
        pad=style["title"]["pad"],
    )

    # Annotate each cell with acc value
    peak_i, peak_j = np.unravel_index(acc_grid.argmax(), acc_grid.shape)
    for i in range(acc_grid.shape[0]):
        for j in range(acc_grid.shape[1]):
            val = acc_grid[i, j]
            # color text white when far from center
            txt_color = "white" if abs(val - baseline_raw) > half * 0.55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color)

    # Highlight peak cell with a yellow box
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((peak_j - 0.5, peak_i - 0.5), 1, 1,
                            fill=False, edgecolor="#FFD600", linewidth=2.5))
    ax.text(peak_j, peak_i + 0.42,
            f"peak {acc_grid[peak_i, peak_j]:.2f}",
            ha="center", va="bottom", fontsize=7, color="#444",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#FFD600", lw=1))

    # Mark baseline cell (nb=10, τ=0)
    base_i = np.where(taus == 0.0)[0][0]
    base_j = np.where(np.array(nbs) == 10)[0][0]
    ax.add_patch(Rectangle((base_j - 0.5, base_i - 0.5), 1, 1,
                            fill=False, edgecolor="#666666", linewidth=1.5,
                            linestyle="--"))

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy (\\%)", fontsize=style["axes"]["label_fontsize"])

    # Footer note about LLaGA targets
    fig.text(0.02, -0.02,
             r"LLaGA-ND-7B 75.98, LLaGA-HO-7B 76.66 (full test, Single Focus)",
             fontsize=7.5, color="#555", ha="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Peak: nb={nbs[peak_j]}, τ={taus[peak_i]} → {acc_grid[peak_i, peak_j]:.2f}")
    print(f"Baseline (nb=10, τ=0): {baseline_raw:.2f}")


if __name__ == "__main__":
    main()
