"""Bar chart: TM-DLM seq=4096 (topo & notopo) vs PubMed baselines.

Best-per-setting on the PubMed test split. TM-DLM numbers are the peaks of
results.md §13 (run tag pubmed_20260502_mcdigit_nonb_seq4k):
  - notopo: 95.40 @ ckpt-248
  - topo:   95.06 @ ckpt-372

Baselines are reproduced from results.md "PubMed -- supervised" (LLaGA paper
Table 1, single-focus).

Output: examples/tmdlm/figures/pubmed_baselines.png
"""
import os
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures", "pubmed_baselines.png")

# (label, accuracy, group). group ∈ {gnn, llm_graph, ours}.
ROWS = [
    ("TM-DLM seq=4096 (notopo)", 95.40, "ours"),
    ("TM-DLM seq=4096 (topo)",   95.06, "ours"),
    ("SAGN",                      95.17, "gnn"),
    ("LLaGA-ND-7B",               95.03, "llm_graph"),
    ("LLaGA-HO-7B",               95.03, "llm_graph"),
    ("NodeFormer",                94.90, "gnn"),
    ("GraphSAGE",                 94.87, "gnn"),
    ("GCN",                       92.96, "gnn"),
    ("GAT",                       92.33, "gnn"),
    ("SGC",                       87.35, "gnn"),
]

COLORS = {
    "gnn":       "#a8a8a8",
    "llm_graph": "#7fa6d6",
    "ours":      "#d95f43",
}
LEGEND = {
    "gnn":       "GNN baselines",
    "llm_graph": "LLM + Graph baselines",
    "ours":      "TM-DLM (ours, seq=4096)",
}

rows_sorted = sorted(ROWS, key=lambda r: r[1])
labels = [r[0] for r in rows_sorted]
accs   = [r[1] for r in rows_sorted]
colors = [COLORS[r[2]] for r in rows_sorted]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
bars = ax.barh(labels, accs, color=colors, edgecolor="black", linewidth=0.5)

xmin = 86
ax.set_xlim(xmin, 96.5)
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.08, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f}", va="center", fontsize=9)

# Reference line: best published baseline (SAGN 95.17).
ax.axvline(95.17, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(95.17, len(labels) - 0.4, " SAGN best published 95.17",
        fontsize=8, color="gray", va="top")

ax.set_xlabel("Test accuracy (%)")
ax.set_title("PubMed node classification: TM-DLM (seq=4096) vs. published baselines")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
ax.legend(handles, LEGEND.values(), loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(axis="x", alpha=0.3, linestyle=":")
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"saved {OUT}")
