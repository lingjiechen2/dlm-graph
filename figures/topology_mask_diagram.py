"""Generate a clear diagram illustrating the Topology Attention Mask."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})

# ── Left: Star Topology Graph ──
ax = axes[0]
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.0)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Star Topology", fontsize=15, fontweight="bold", pad=12)

# Node positions
target_pos = (0, 0.3)
nb_positions = [(-1.5, -1.4), (0, -1.6), (1.5, -1.4)]
nb_labels = ["Neighbor 1", "Neighbor 2", "Neighbor 3"]

# Draw edges (bidirectional arrows)
for pos in nb_positions:
    ax.annotate(
        "",
        xy=(pos[0] * 0.55, target_pos[1] + (pos[1] - target_pos[1]) * 0.45),
        xytext=(pos[0] * 0.82, pos[1] + (target_pos[1] - pos[1]) * 0.35),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.8),
    )
    ax.annotate(
        "",
        xy=(pos[0] * 0.82, pos[1] + (target_pos[1] - pos[1]) * 0.35),
        xytext=(pos[0] * 0.55, target_pos[1] + (pos[1] - target_pos[1]) * 0.45),
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.8),
    )

# Draw "no connection" dashed lines between neighbors
for i in range(len(nb_positions)):
    for j in range(i + 1, len(nb_positions)):
        p1, p2 = nb_positions[i], nb_positions[j]
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2 - 0.15
        ax.plot(
            [p1[0], p2[0]],
            [p1[1] - 0.15, p2[1] - 0.15],
            linestyle="--",
            color="#ccc",
            lw=1.2,
            zorder=0,
        )
        ax.text(mid_x, mid_y - 0.18, "✗", ha="center", va="center", fontsize=13, color="#d44")

# Target node
circle_t = plt.Circle(target_pos, 0.42, color="#4A90D9", ec="black", lw=2, zorder=5)
ax.add_patch(circle_t)
ax.text(
    target_pos[0], target_pos[1], "Target", ha="center", va="center",
    fontsize=11, fontweight="bold", color="white", zorder=6,
)
ax.text(
    target_pos[0], target_pos[1] + 0.7, "attends to ALL",
    ha="center", va="center", fontsize=9, color="#4A90D9", style="italic",
)

# Neighbor nodes
for pos, label in zip(nb_positions, nb_labels):
    circle_n = plt.Circle(pos, 0.38, color="#F5A623", ec="black", lw=2, zorder=5)
    ax.add_patch(circle_n)
    ax.text(
        pos[0], pos[1], label.split()[1],
        ha="center", va="center", fontsize=10, fontweight="bold", zorder=6,
    )
    ax.text(
        pos[0], pos[1] - 0.62, f"attends to\nself + target",
        ha="center", va="top", fontsize=8, color="#666", style="italic",
    )

# Legend
legend_elements = [
    mpatches.Patch(facecolor="#4A90D9", edgecolor="black", label="Target node"),
    mpatches.Patch(facecolor="#F5A623", edgecolor="black", label="Neighbor node"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

# ── Right: Attention Mask Matrix ──
ax2 = axes[1]
ax2.set_title("Attention Mask Matrix", fontsize=15, fontweight="bold", pad=12)

labels = ["Target", "Nb 1", "Nb 2", "Nb 3"]
n = len(labels)

# 1 = can attend, 0 = blocked
mask = np.array(
    [
        [1, 1, 1, 1],  # Target attends to all
        [1, 1, 0, 0],  # Nb1 attends to Target + self
        [1, 0, 1, 0],  # Nb2 attends to Target + self
        [1, 0, 0, 1],  # Nb3 attends to Target + self
    ]
)

# Custom colormap: green for attend, light gray for blocked
colors = np.zeros((*mask.shape, 4))
for i in range(n):
    for j in range(n):
        if mask[i, j] == 1:
            colors[i, j] = [0.29, 0.56, 0.85, 0.85]  # blue = attend
        else:
            colors[i, j] = [0.93, 0.93, 0.93, 1.0]  # light gray = blocked

ax2.imshow(colors, aspect="equal")

# Add text annotations
for i in range(n):
    for j in range(n):
        text = "✓" if mask[i, j] == 1 else "✗"
        color = "white" if mask[i, j] == 1 else "#bbb"
        ax2.text(j, i, text, ha="center", va="center", fontsize=18, fontweight="bold", color=color)

# Labels
ax2.set_xticks(range(n))
ax2.set_xticklabels(labels, fontsize=11)
ax2.set_yticks(range(n))
ax2.set_yticklabels(labels, fontsize=11)
ax2.set_xlabel("Key (attends to →)", fontsize=12, labelpad=8)
ax2.set_ylabel("Query ↓", fontsize=12, labelpad=8)

# Grid
for i in range(n + 1):
    ax2.axhline(i - 0.5, color="white", lw=2)
    ax2.axvline(i - 0.5, color="white", lw=2)

# Annotation
ax2.text(
    1.5, 4.3, "Blue = can attend    Gray = blocked",
    ha="center", va="center", fontsize=10, color="#555",
)

plt.tight_layout(pad=2.0)
plt.savefig(
    "/home/lingjie7/auto-research/projects/dlm-graph/figures/topology_mask_diagram.png",
    dpi=200, bbox_inches="tight", facecolor="white",
)
plt.savefig(
    "/home/lingjie7/auto-research/projects/dlm-graph/figures/topology_mask_diagram.pdf",
    bbox_inches="tight", facecolor="white",
)
print("Saved to figures/topology_mask_diagram.png and .pdf")
