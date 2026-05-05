"""CDF of neighbor density across cora / pubmed / ogbn-arxiv / ogbn-products.

Two panels in one figure:
  Left  - raw 1-hop degree CDF (log-x, since arxiv max=1251 vs cora max=168)
  Right - sampled total neighbors CDF after _sample_khop_neighbors(
          max_neighbors_per_hop=10, max_hops=2)  - bounded by 20

Methodology matches results.md s15: 2000-node random sample of each
dataset's train split (or full split if smaller), seed=42.

Output: analysis/figures/neighbor_density_cdf.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dllm.data.graph import _sample_khop_neighbors, DATASET_CONFIGS
from dllm.data.datasets import LOADERS as _DATA_LOADERS

DATASETS = ["cora", "pubmed", "ogbn-arxiv", "ogbn-products"]
N_SAMPLE = 2000
SEED = 42

# Distinct color per dataset (avoid overlap with the blue/orange topo palette).
COLORS = {
    "cora":          "#2166AC",
    "pubmed":        "#D55E00",
    "ogbn-arxiv":    "#117733",
    "ogbn-products": "#882255",
}


def compute_degrees(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw_deg, sampled_nb_count) per sampled train node."""
    print(f"[{name}] loading...")
    _, adj, _, split_ids = _DATA_LOADERS[name](DATASET_CONFIGS[name], "train", SEED)
    train_nodes = list(split_ids)
    rng = random.Random(SEED)
    if len(train_nodes) > N_SAMPLE:
        train_nodes = rng.sample(train_nodes, N_SAMPLE)
    print(f"[{name}] sampling {len(train_nodes)} nodes...")

    raw_deg = np.array([len(adj.get(n, [])) for n in train_nodes], dtype=np.int64)
    sampled = np.zeros(len(train_nodes), dtype=np.int64)
    for i, n in enumerate(train_nodes):
        nb_ids, _ = _sample_khop_neighbors(adj, n, 10, 2, rng)
        sampled[i] = len(nb_ids)
    return raw_deg, sampled


def cdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF as smooth quantile curve.

    Apply tiny uniform jitter ([-0.5, 0.5]) to the integer-valued degrees so
    ties break and the resulting quantile sequence is strictly monotone. With
    201 evenly spaced quantiles this yields a continuous, jitter-free curve
    that preserves the underlying CDF shape (the +/-0.5 jitter is below the
    plotted axis precision in both panels).
    """
    rng = np.random.default_rng(0)
    jittered = values.astype(np.float64) + rng.uniform(-0.5, 0.5, size=values.shape)
    qs = np.linspace(0.0, 1.0, 201)
    vs = np.quantile(jittered, qs, method="linear")
    return vs, qs


def load_style(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def apply_rcparams(style: dict) -> None:
    f = style["font"]
    plt.rcParams.update({
        "text.usetex": f.get("usetex", False),
        "text.latex.preamble": f.get("latex_preamble", ""),
        "font.family": f["family"],
        "axes.spines.top": style["axes"]["spines_top"],
        "axes.spines.right": style["axes"]["spines_right"],
        "xtick.labelsize": style["axes"]["tick_labelsize"],
        "ytick.labelsize": style["axes"]["tick_labelsize"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="analysis/plot_style.json")
    parser.add_argument("--out",   default="analysis/figures/neighbor_density_cdf.png")
    args = parser.parse_args()

    style = load_style(Path(args.style))
    apply_rcparams(style)

    data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in DATASETS:
        data[name] = compute_degrees(name)

    fig, (ax_raw, ax_smp) = plt.subplots(
        1, 2, figsize=(12, 5.0), dpi=style["figure"]["dpi"]
    )

    # Axes flipped (CDF on x, degree on y) and rendered as smooth lines.
    for name in DATASETS:
        raw, smp = data[name]
        c = COLORS[name]

        v, q = cdf_xy(raw)
        ax_raw.plot(q, v, color=c, linewidth=2.0, label=name)

        v, q = cdf_xy(smp)
        ax_smp.plot(q, v, color=c, linewidth=2.0, label=name)

    ax_raw.set_yscale("symlog", linthresh=1)
    ax_raw.set_ylim(0, 2000)
    ax_raw.set_xlim(0, 1)
    ax_raw.set_xlabel(r"Empirical CDF",
                      fontsize=style["axes"]["label_fontsize"])
    ax_raw.set_ylabel("Raw 1-hop degree (log scale)",
                      fontsize=style["axes"]["label_fontsize"])
    ax_raw.set_title("Raw 1-hop degree", fontsize=style["title"]["fontsize"],
                     pad=style["title"]["pad"])
    ax_raw.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_raw.grid(axis="both", linestyle="--", alpha=0.3)
    ax_raw.legend(frameon=False, fontsize=9, loc="upper left")

    ax_smp.set_ylim(0, 21)
    ax_smp.set_yticks(range(0, 22, 2))
    ax_smp.set_xlim(0, 1)
    ax_smp.set_xlabel(r"Empirical CDF",
                      fontsize=style["axes"]["label_fontsize"])
    ax_smp.set_ylabel(r"Sampled neighbors (\texttt{nb=10, hop=2})",
                      fontsize=style["axes"]["label_fontsize"])
    ax_smp.set_title("Sampled neighbors fed to model",
                     fontsize=style["title"]["fontsize"],
                     pad=style["title"]["pad"])
    ax_smp.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_smp.grid(axis="both", linestyle="--", alpha=0.3)
    ax_smp.axhline(20, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_smp.text(0.02, 20, "cap=20", fontsize=8, color="gray", va="bottom")
    ax_smp.legend(frameon=False, fontsize=9, loc="upper left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
