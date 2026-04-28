"""
Plot Cora and PubMed neighbor-count sweep results.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/figures/plot_cora_pubmed_results.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        # LaTeX-like typography without requiring external latex runtime.
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)

NB_VALUES = [1, 3, 5, 10, 20]
LINE_WIDTH = 2.8
MARKER_SIZE = 5.0
BAR_HEIGHT = 0.42
HOPS_TO_PLOT = [1, 3]

def _plot_nb_sweep(
    ax,
    title: str,
    data_false: dict[int, list[float]],
    data_true: dict[int, list[float]],
    accent_colors: list[str],
) -> None:
    for idx, hop in enumerate(HOPS_TO_PLOT):
        ax.plot(
            NB_VALUES,
            data_false[hop],
            color=accent_colors[idx],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=f"h={hop}, topo=False",
        )

    gray_palette = ["#808080", "#b2b2b2"]
    for idx, hop in enumerate(HOPS_TO_PLOT):
        ax.plot(
            NB_VALUES,
            data_true[hop],
            color=gray_palette[idx],
            linewidth=LINE_WIDTH,
            linestyle="--",
            marker="o",
            markersize=MARKER_SIZE,
            label=f"h={hop}, topo=True",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Neighbor Count (nb)", fontsize=11)
    ax.set_ylabel("Accuracy Strict (%)", fontsize=11)
    ax.set_xticks(NB_VALUES)
    ax.grid(axis="both", linestyle="--", alpha=0.28)
    ax.legend(fontsize=8.5, ncol=2, frameon=False, loc="best")


def _plot_bar_results(ax, title: str, rows: list[tuple[str, float]], ours_color: str) -> None:
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    methods = [x[0] for x in rows_sorted]
    scores = [x[1] for x in rows_sorted]
    colors = [ours_color if m.lower().startswith("ours") else "#c7c7c7" for m in methods]

    y = list(range(len(methods)))
    ax.barh(
        y,
        scores,
        height=BAR_HEIGHT,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    x_min = min(scores) - 2.0
    x_max = max(scores) + 2.0
    ax.set_xlim(x_min, x_max)

    for yi, val in zip(y, scores):
        ax.text(val + 0.12, yi, f"{val:.2f}", va="center", ha="left", fontsize=8.5)


def main() -> None:
    # Source: /home/lingjie7/auto-research/projects/dlm-graph/README.md
    # Neighbor Count Sweep (use_topology_mask=False)
    cora_false = {
        1: [60.52, 60.33, 61.62, 61.44, 61.44],
        3: [57.93, 63.84, 65.13, 64.21, 64.76],
    }
    pubmed_false = {
        1: [74.77, 80.98, 81.68, 81.78, 82.08],
        3: [85.19, 88.89, 88.99, 88.89, 90.29],
    }

    # Neighbor Count Sweep (use_topology_mask=True)
    cora_true = {
        1: [60.89, 59.41, 60.52, 60.33, 60.33],
        3: [57.01, 61.44, 62.55, 62.55, 62.73],
    }
    pubmed_true = {
        1: [74.97, 76.78, 77.58, 77.88, 77.98],
        3: [77.08, 81.88, 82.78, 82.28, 82.68],
    }

    # Previous bar-chart setting (overall method comparison).
    cora_bar_rows = [
        ("GCN + LLM Emb", 88.15),
        ("TAPE", 88.05),
        ("LLaGA", 87.55),
        ("GraphSAGE", 87.44),
        ("GCN", 87.41),
        ("Ours: SFT (1-hop + topo mask)", 84.87),
        ("Ours: SFT (1-hop, no topo mask)", 84.13),
        ("Ours: SFT (2-hop + topo mask)", 83.95),
        ("Ours: SFT (2-hop, no topo mask)", 84.50),
        ("RoBERTa-355M", 83.17),
        ("Ours: Frozen MC", 62.73),
    ]
    pubmed_bar_rows = [
        ("RoBERTa-355M", 94.84),
        ("TAPE", 93.00),
        ("Ours: Frozen (no_topo)", 90.29),
        ("GCN", 89.01),
        ("Ours: Frozen (with_topo)", 88.69),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    _plot_nb_sweep(
        axes[0],
        "Cora Neighbor Sweep (topo=False/True)",
        cora_false,
        cora_true,
        accent_colors=["#2a9d8f", "#52b7a5"],
    )
    _plot_nb_sweep(
        axes[1],
        "PubMed Neighbor Sweep (topo=False/True)",
        pubmed_false,
        pubmed_true,
        accent_colors=["#e76f51", "#f4a261"],
    )

    fig.suptitle(
        "Neighbor-Count Sweep on Cora and PubMed",
        fontsize=16,
        fontweight="bold",
    )

    png_path = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_pubmed_results_comparison.png"
    pdf_path = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_pubmed_results_comparison.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)

    # Split image: Cora only
    fig_cora, ax_cora = plt.subplots(1, 1, figsize=(9.2, 5.8), constrained_layout=True)
    _plot_nb_sweep(
        ax_cora,
        "Cora Neighbor Sweep (topo=False/True)",
        cora_false,
        cora_true,
        accent_colors=["#2a9d8f", "#52b7a5"],
    )
    cora_png = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_results_only.png"
    cora_pdf = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_results_only.pdf"
    fig_cora.savefig(cora_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig_cora.savefig(cora_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {cora_png}")
    print(f"Saved: {cora_pdf}")
    plt.close(fig_cora)

    # Split image: PubMed only
    fig_pubmed, ax_pubmed = plt.subplots(1, 1, figsize=(9.2, 5.8), constrained_layout=True)
    _plot_nb_sweep(
        ax_pubmed,
        "PubMed Neighbor Sweep (topo=False/True)",
        pubmed_false,
        pubmed_true,
        accent_colors=["#e76f51", "#f4a261"],
    )
    pubmed_png = "/home/lingjie7/auto-research/projects/dlm-graph/figures/pubmed_results_only.png"
    pubmed_pdf = "/home/lingjie7/auto-research/projects/dlm-graph/figures/pubmed_results_only.pdf"
    fig_pubmed.savefig(pubmed_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig_pubmed.savefig(pubmed_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pubmed_png}")
    print(f"Saved: {pubmed_pdf}")
    plt.close(fig_pubmed)

    # Previous style bar charts
    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    _plot_bar_results(
        axes_bar[0], "Cora Results (Bar)", cora_bar_rows, ours_color="#2a9d8f"
    )
    _plot_bar_results(
        axes_bar[1], "PubMed Results (Bar)", pubmed_bar_rows, ours_color="#e76f51"
    )
    fig_bar.suptitle(
        "DLM-Graph Results: Ours Highlighted, Others in Gray",
        fontsize=16,
        fontweight="bold",
    )
    bar_png = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_pubmed_results_comparison_bar.png"
    bar_pdf = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_pubmed_results_comparison_bar.pdf"
    fig_bar.savefig(bar_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig_bar.savefig(bar_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {bar_png}")
    print(f"Saved: {bar_pdf}")
    plt.close(fig_bar)

    fig_cora_bar, ax_cora_bar = plt.subplots(
        1, 1, figsize=(10, 7), constrained_layout=True
    )
    _plot_bar_results(
        ax_cora_bar, "Cora Results (Bar)", cora_bar_rows, ours_color="#2a9d8f"
    )
    cora_bar_png = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_results_only_bar.png"
    cora_bar_pdf = "/home/lingjie7/auto-research/projects/dlm-graph/figures/cora_results_only_bar.pdf"
    fig_cora_bar.savefig(cora_bar_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig_cora_bar.savefig(cora_bar_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {cora_bar_png}")
    print(f"Saved: {cora_bar_pdf}")
    plt.close(fig_cora_bar)

    fig_pubmed_bar, ax_pubmed_bar = plt.subplots(
        1, 1, figsize=(9, 5), constrained_layout=True
    )
    _plot_bar_results(
        ax_pubmed_bar, "PubMed Results (Bar)", pubmed_bar_rows, ours_color="#e76f51"
    )
    pubmed_bar_png = "/home/lingjie7/auto-research/projects/dlm-graph/figures/pubmed_results_only_bar.png"
    pubmed_bar_pdf = "/home/lingjie7/auto-research/projects/dlm-graph/figures/pubmed_results_only_bar.pdf"
    fig_pubmed_bar.savefig(
        pubmed_bar_png, dpi=220, bbox_inches="tight", facecolor="white"
    )
    fig_pubmed_bar.savefig(pubmed_bar_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {pubmed_bar_png}")
    print(f"Saved: {pubmed_bar_pdf}")
    plt.close(fig_pubmed_bar)


if __name__ == "__main__":
    main()
