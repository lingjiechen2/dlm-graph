"""Bar chart: TM-DLM seq=4096 in-domain vs. cross-domain transfer.

All numbers are best-per-setting from results.md, restricted to seq=4096 runs:

  Cora test (cora-trained, §11 run tag cora_20260502_mcdigit_nonb_seq4k):
    - in-domain notopo: 89.67 @ ckpt-221
    - in-domain topo:   88.56 @ ckpt-187/238 (tied)

  PubMed test, in-domain (§13 run tag pubmed_20260502_mcdigit_nonb_seq4k):
    - in-domain notopo: 95.40 @ ckpt-248
    - in-domain topo:   95.06 @ ckpt-372

  PubMed test, cross-domain (§11: cora-only seq=4096 ckpts evaluated on
                             pubmed test n=1000):
    - cross-domain notopo: 91.20 @ ckpt-136/153 (tied)
    - cross-domain topo:   90.70 @ ckpt-272/340 (tied)

Output: examples/tmdlm/figures/cross_dataset.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures", "cross_dataset.png")

# (panel-label, [(setting-label, notopo, topo, is_cross)])
DATA = [
    (
        "Cora test",
        [
            ("In-domain\n(cora-train)", 89.67, 88.56, False),
        ],
    ),
    (
        "PubMed test",
        [
            ("In-domain\n(pubmed-train)", 95.40, 95.06, False),
            ("Cross-domain\n(cora-train only)", 91.20, 90.70, True),
        ],
    ),
]

C_NOTOPO   = "#d95f43"
C_TOPO     = "#3a7d99"
HATCH_X    = "//"  # cross-domain bars get a hatch
WIDTH      = 0.36

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6),
                         gridspec_kw={"width_ratios": [1, 2]})

for ax, (panel, rows) in zip(axes, DATA):
    n = len(rows)
    x = np.arange(n)
    notopo_vals = [r[1] for r in rows]
    topo_vals   = [r[2] for r in rows]
    cross_flags = [r[3] for r in rows]

    b1 = ax.bar(x - WIDTH / 2, notopo_vals, WIDTH, color=C_NOTOPO,
                edgecolor="black", linewidth=0.5, label="notopo")
    b2 = ax.bar(x + WIDTH / 2, topo_vals, WIDTH, color=C_TOPO,
                edgecolor="black", linewidth=0.5, label="topo")
    # Hatch the cross-domain bars.
    for i, is_cross in enumerate(cross_flags):
        if is_cross:
            b1[i].set_hatch(HATCH_X)
            b2[i].set_hatch(HATCH_X)

    for bars, vals in [(b1, notopo_vals), (b2, topo_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                    f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=9)
    ax.set_title(panel)
    ax.set_ylim(85, 97)
    ax.set_ylabel("Test accuracy (%)") if ax is axes[0] else None
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)

# Shared legend (notopo / topo / hatch = cross-domain).
hatch_handle = plt.Rectangle((0, 0), 1, 1, facecolor="white",
                             edgecolor="black", hatch=HATCH_X)
notopo_h = plt.Rectangle((0, 0), 1, 1, facecolor=C_NOTOPO, edgecolor="black")
topo_h   = plt.Rectangle((0, 0), 1, 1, facecolor=C_TOPO,   edgecolor="black")
fig.legend(
    [notopo_h, topo_h, hatch_handle],
    ["notopo", "topo", "cross-domain (no target-dataset training)"],
    loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02),
    frameon=False,
)
fig.suptitle("TM-DLM seq=4096: in-domain vs. cross-domain transfer",
             fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved {OUT}")
