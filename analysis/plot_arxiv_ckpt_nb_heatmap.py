"""
Heatmap of arxiv §21 r128 RAW accuracy at N=5000 across (ckpt × max_neighbors_per_hop).
Single panel over the Phase 6 dense 4×4 grid.

Output: analysis/figures/arxiv_ckpt_nb_heatmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_utils import apply_rcparams, load_jsonl, load_style


def build_n5000_grid(eval_root: Path):
    """Phase 6 grid (4 ckpts × 4 nb at N=5000)."""
    grid = {}
    for d in load_jsonl(eval_root / "phase6_5k_nb_sweep.jsonl"):
        exp = d.get("experiment", "")
        if "phase6-" not in exp:
            continue
        tag = exp.replace("phase6-", "")  # e.g., 1845_nb12
        try:
            ck, nb_str = tag.split("_nb")
            nb = int(nb_str)
        except ValueError:
            continue
        a = d.get("accuracy")
        if a is None:
            continue
        grid[(ck, nb)] = float(a)
    return grid


def render_heatmap(ax, grid, ckpts, nbs, title, vmin=None, vmax=None):
    M = np.full((len(ckpts), len(nbs)), np.nan)
    for i, ck in enumerate(ckpts):
        for j, nb in enumerate(nbs):
            if (ck, nb) in grid:
                M[i, j] = grid[(ck, nb)]

    finite = M[np.isfinite(M)]
    if len(finite) == 0:
        ax.set_title(title + " (no data)")
        return None
    if vmin is None:
        vmin = finite.min()
    if vmax is None:
        vmax = finite.max()
    # Diverging centered around the SFT-default cell if it exists
    default_val = grid.get(("1845", 10))
    if default_val is not None:
        half = max(default_val - vmin, vmax - default_val) * 1.05
        vmin, vmax = default_val - half, default_val + half

    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(nbs)))
    ax.set_xticklabels([str(n) for n in nbs])
    ax.set_yticks(range(len(ckpts)))
    ax.set_yticklabels([f"ckpt-{c}" for c in ckpts])

    # Cell annotations
    for i in range(len(ckpts)):
        for j in range(len(nbs)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=9, color="#999")
            else:
                # white text far from center
                txt_color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.30 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color)


    ax.set_title(title, fontsize=11, pad=6)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--style", default="analysis/plot_style.json")
    p.add_argument("--out", default="analysis/figures/arxiv_ckpt_nb_heatmap.png")
    args = p.parse_args()

    style = load_style(Path(args.style))
    apply_rcparams(style)

    eval_root = Path("analysis/postprocess_arxiv_r128/eval_jsonl")
    g5k = build_n5000_grid(eval_root)

    ckpts = ["1640", "1845", "2042", "final"]
    nbs_5k = [10, 12, 15, 30]

    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=style["figure"]["dpi"])

    im = render_heatmap(
        ax, g5k, ckpts, nbs_5k,
        "",  # no per-axes title
    )

    if im is not None:
        fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015, aspect=14).set_label(
            "Acc (\\%)", fontsize=9)

    ax.set_xlabel(r"\texttt{max\_neighbors\_per\_hop}", fontsize=style["axes"]["label_fontsize"])
    ax.set_ylabel("checkpoint", fontsize=style["axes"]["label_fontsize"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
