"""Bar chart: self / joint / cross-domain eval on Cora and PubMed test sets.

For each test set and each train regime we report best-per-setting
notopo / topo numbers from results.md:

  Cora test:
    - self  (§1, single-cora seq=2k):              notopo 90.77 / topo 90.04
    - joint (§7, cora+pubmed merged-bal):          notopo 90.77 / topo 90.96
    - cross (pubmed-only -> cora):                 [no data]

  PubMed test:
    - self  (§13, single-pubmed seq=4k):           notopo 95.40 / topo 95.06
    - joint (§7, cora+pubmed merged-bal):          notopo 95.28 / topo 94.93
    - cross (§11, cora-only seq=4k -> pubmed n=1k): notopo 91.20 / topo 90.70

Output: examples/tmdlm/figures/cross_vs_self_eval.png
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures", "cross_vs_self_eval.png")
STYLE_PATH = os.path.join(os.path.dirname(__file__), "plot_style.json")

with open(STYLE_PATH) as _f:
    _style = json.load(_f)
_font = _style["font"]
plt.rcParams.update({
    "text.usetex": _font.get("usetex", False),
    "text.latex.preamble": _font.get("latex_preamble", ""),
    "font.family": _font["family"],
    "axes.spines.top": _style["axes"]["spines_top"],
    "axes.spines.right": _style["axes"]["spines_right"],
    "xtick.labelsize": _style["axes"]["tick_labelsize"],
    "ytick.labelsize": _style["axes"]["tick_labelsize"],
})

# (group-label, notopo, topo). None = data unavailable.
DATA = {
    "Cora test": [
        ("Self\n(cora-train)",                90.77, 90.04),
        ("Joint\n(cora+pubmed)",              90.77, 90.96),
        ("Cross\n(pubmed-train)",             None,  None),
    ],
    "PubMed test": [
        ("Self\n(pubmed-train)",              95.40, 95.06),
        ("Joint\n(cora+pubmed)",              95.28, 94.93),
        ("Cross\n(cora-train, n=1000)",       91.20, 90.70),
    ],
}

C_NOTOPO = "#d95f43"
C_TOPO   = "#3a7d99"
WIDTH    = 0.36

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

for ax, (panel, rows) in zip(axes, DATA.items()):
    n = len(rows)
    x = np.arange(n)
    notopo_vals = [r[1] if r[1] is not None else 0 for r in rows]
    topo_vals   = [r[2] if r[2] is not None else 0 for r in rows]

    b1 = ax.bar(x - WIDTH / 2, notopo_vals, WIDTH, color=C_NOTOPO,
                edgecolor="black", linewidth=0.5, label="notopo")
    b2 = ax.bar(x + WIDTH / 2, topo_vals, WIDTH, color=C_TOPO,
                edgecolor="black", linewidth=0.5, label="topo")

    for i, (label, no_v, to_v) in enumerate(rows):
        if no_v is None:
            # Annotate "no data" centered on the empty group.
            ax.text(x[i], 86.0, "no eval\nrun yet", ha="center", va="center",
                    fontsize=10, color="gray", style="italic",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4",
                             ec="gray", lw=0.7))
            # Hide the zero-height bars for this group.
            b1[i].set_visible(False)
            b2[i].set_visible(False)
            continue
        ax.text(x[i] - WIDTH / 2, no_v + 0.18, f"{no_v:.2f}",
                ha="center", fontsize=9)
        ax.text(x[i] + WIDTH / 2, to_v + 0.18, f"{to_v:.2f}",
                ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title(panel, fontsize=12)
    ax.set_ylim(85, 97)
    ax.set_ylabel(r"Test accuracy (\%)") if ax is axes[0] else None
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

# Shared legend.
notopo_h = plt.Rectangle((0, 0), 1, 1, facecolor=C_NOTOPO, edgecolor="black")
topo_h   = plt.Rectangle((0, 0), 1, 1, facecolor=C_TOPO,   edgecolor="black")
fig.legend([notopo_h, topo_h], ["notopo", "topo"],
           loc="upper center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 1.02), frameon=False)
fig.suptitle("Self / Joint / Cross-domain evaluation across Cora and PubMed",
             fontsize=12, y=1.07)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}")
