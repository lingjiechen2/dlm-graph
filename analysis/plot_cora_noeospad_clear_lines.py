"""Rebuild the Cora no-eos-pad checkpoint line plot with epoch x-axis and README baselines.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/plot_cora_noeospad_clear_lines.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_CSV = Path(
    "/home/lingjie7/auto-research/projects/dlm-graph/summaries/"
    "cora_noeospad_allckpts_eval_gpu01_20260425_164435/summary.csv"
)
OUT_DIR = SUMMARY_CSV.parent
PNG_OUT = OUT_DIR / "cora_noeospad_eval_clear_lines.png"
PDF_OUT = OUT_DIR / "cora_noeospad_eval_clear_lines.pdf"

# Current Cora baselines in README.md.
BASELINES = {
    "LLaGA-HO-7B": 89.22,
    "SAGN": 89.19,
    "GCN": 88.93,
    "RoBERTa-355M": 83.17,
}


def checkpoint_to_epoch(checkpoint_name: str) -> float:
    if checkpoint_name == "checkpoint-final":
        return 20.0
    step = int(checkpoint_name.replace("checkpoint-", ""))
    return step / 102.0


def checkpoint_order_key(checkpoint_name: str) -> tuple[float, int]:
    epoch = checkpoint_to_epoch(checkpoint_name)
    is_final = 1 if checkpoint_name == "checkpoint-final" else 0
    return (epoch, is_final)


def prepare_run_df(run_df: pd.DataFrame, eval_type: str, metric_col: str) -> pd.DataFrame:
    sub = run_df[run_df["eval_type"] == eval_type].copy()
    sub["metric"] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub["metric"].notna()].copy()
    sub["epoch"] = sub["checkpoint_name"].map(checkpoint_to_epoch)
    sub["is_final"] = sub["checkpoint_name"].eq("checkpoint-final").astype(int)
    sub = sub.sort_values(["epoch", "is_final"])
    # `checkpoint-2040` and `checkpoint-final` both map to epoch 20.
    # Keep `checkpoint-final` when both exist.
    sub = sub.drop_duplicates(subset=["epoch"], keep="last")
    return sub


def main() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["return_code"] == 0].copy()

    run_order = [
        ("No-Topo", False),
        ("Topo", True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.6), dpi=220, sharey=True)

    color_logit = "#4C4C4C"
    color_infill = "#D55E00"
    baseline_colors = ["#9A9A9A", "#B0B0B0", "#C4C4C4", "#D6D6D6"]

    for ax, (title, topo_flag) in zip(axes, run_order):
        run_df = df[df["use_topology_mask"] == topo_flag].copy()
        if run_df.empty:
            continue

        logit_df = prepare_run_df(run_df, "logit", "accuracy")
        infill_df = prepare_run_df(run_df, "infill", "accuracy_lenient")

        ax.plot(
            logit_df["epoch"],
            logit_df["metric"],
            marker="o",
            linewidth=2.2,
            markersize=4.4,
            color=color_logit,
            label="Logit",
            zorder=3,
        )
        ax.plot(
            infill_df["epoch"],
            infill_df["metric"],
            marker="o",
            linewidth=2.2,
            markersize=4.4,
            color=color_infill,
            label="Infill",
            zorder=3,
        )

        for (label, value), baseline_color in zip(BASELINES.items(), baseline_colors):
            ax.axhline(
                value,
                color=baseline_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
                label=label,
                zorder=1,
            )

        ax.set_title(f"Cora {title}", fontsize=18, fontweight="bold", pad=6)
        ax.set_xlabel("Training Epoch", fontsize=18)
        ax.set_xticks(list(range(2, 21, 2)))
        ax.set_xlim(1.2, 20.8)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=28, labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

    axes[0].set_ylabel("Accuracy (%)", fontsize=18)
    axes[0].set_ylim(74.0, 92.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=False,
        fontsize=16.5,
        columnspacing=1.15,
        handletextpad=0.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(PNG_OUT, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {PDF_OUT}")


if __name__ == "__main__":
    main()
