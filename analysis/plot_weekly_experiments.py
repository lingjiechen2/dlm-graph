"""Plot weekly experiment summaries for Cora and PubMed.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/plot_weekly_experiments.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
OUT_DIR = ROOT / "figures" / "weekly_experiment_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORA_LABEL_MERGED = ROOT / "summaries" / "cora_label_ablation_merged" / "merged_results.csv"
PUBMED_MASK_SWEEP = ROOT / "summaries" / "pubmed_frozen_mask_sweep_3gpu_20260426_041215" / "summary.csv"


def setup_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False


def parse_step(checkpoint_name: str) -> int:
    if checkpoint_name == "checkpoint-final":
        return 10**9
    return int(checkpoint_name.replace("checkpoint-", ""))


def short_run_name(run_label: str) -> str:
    return (
        run_label.replace("-base-lora-20260418_185914", "")
        .replace("-ckpt", "")
        .replace("na-na-", "")
        .replace("2hop-na-fullattn", "2hop-fullattn")
    )


def plot_cora_first_last_neighbor_ablation() -> list[Path]:
    df = pd.read_csv(CORA_LABEL_MERGED).copy()
    df = df[df["return_code"] == 0].copy()
    df["run_label"] = df["run_label"].map(short_run_name)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["metric"] = pd.to_numeric(df["metric"], errors="coerce")

    rows = []
    for (run_label, eval_type, include_labels), sub in df.groupby(
        ["run_label", "eval_type", "include_neighbor_labels"]
    ):
        sub = sub.sort_values("step")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        last_step = int(last["step"])
        rows.append(
            {
                "run_label": run_label,
                "eval_type": eval_type,
                "include_neighbor_labels": include_labels,
                "stage": "trained 1 epoch",
                "metric": float(first["metric"]),
                "legend_stage": "trained 1 epoch",
            }
        )
        rows.append(
            {
                "run_label": run_label,
                "eval_type": eval_type,
                "include_neighbor_labels": include_labels,
                "stage": f"sft-{last_step} epoch",
                "metric": float(last["metric"]),
                "legend_stage": "late checkpoint",
            }
        )

    plot_df = pd.DataFrame(rows)
    run_order = sorted(plot_df["run_label"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=220, sharey=True)

    color_map = {
        ("trained 1 epoch", False): "#9C9C9C",
        ("trained 1 epoch", True): "#B8D5F1",
        ("late checkpoint", False): "#D55E00",
        ("late checkpoint", True): "#1F77B4",
    }
    label_map = {
        ("trained 1 epoch", False): "Trained 1 Epoch, label off",
        ("trained 1 epoch", True): "Trained 1 Epoch, label on",
        ("late checkpoint", False): "SFT-N Epoch, label off",
        ("late checkpoint", True): "SFT-N Epoch, label on",
    }

    width = 0.18
    offsets = {
        ("trained 1 epoch", False): -1.5 * width,
        ("trained 1 epoch", True): -0.5 * width,
        ("late checkpoint", False): 0.5 * width,
        ("late checkpoint", True): 1.5 * width,
    }

    for ax, eval_type, title in zip(axes, ["logit", "infill"], ["Logit", "Infill"]):
        sub = plot_df[plot_df["eval_type"] == eval_type].copy()
        x = list(range(len(run_order)))
        for key in [
            ("trained 1 epoch", False),
            ("trained 1 epoch", True),
            ("late checkpoint", False),
            ("late checkpoint", True),
        ]:
            legend_stage, include_labels = key
            vals = []
            for run in run_order:
                row = sub[
                    (sub["run_label"] == run)
                    & (sub["legend_stage"] == legend_stage)
                    & (sub["include_neighbor_labels"] == include_labels)
                ]
                vals.append(float(row.iloc[0]["metric"]) if not row.empty else math.nan)
            ax.bar(
                [xi + offsets[key] for xi in x],
                vals,
                width=width,
                color=color_map[key],
                label=label_map[key],
            )
        ax.set_title(f"Cora {title}", fontsize=16, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(run_order, rotation=22, ha="right", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    png = OUT_DIR / "cora_first_last_neighbor_ablation.png"
    pdf = OUT_DIR / "cora_first_last_neighbor_ablation.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_pubmed_mask_sweep() -> list[Path]:
    df = pd.read_csv(PUBMED_MASK_SWEEP).copy()
    df = df[df["return_code"] == 0].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=220)

    for tag, color in [("old", "#6E6E6E"), ("new", "#D55E00")]:
        sub = df[df["eval_tag"] == tag].sort_values("mask_tokens")
        ax.plot(
            sub["mask_tokens"],
            sub["accuracy"],
            marker="o",
            linewidth=2.6,
            markersize=5.4,
            color=color,
            label=tag.upper(),
        )
    ax.set_title("PubMed Frozen: Mask Token Sweep", fontsize=16, fontweight="bold")
    ax.set_xlabel("Mask Tokens", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_xticks(sorted(df["mask_tokens"].unique()))
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)

    png = OUT_DIR / "pubmed_frozen_mask_sweep.png"
    pdf = OUT_DIR / "pubmed_frozen_mask_sweep.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


NB_VALUES = [1, 3, 5, 10, 20]
HOP_VALUES = [1, 2, 3]

SWEEP = {
    "cora": {
        False: {
            1: [60.52, 60.33, 61.62, 61.44, 61.44],
            2: [57.93, 63.84, 65.13, 64.21, 64.76],
            3: [57.93, 63.84, 65.13, 64.21, 64.76],
        },
        True: {
            1: [60.89, 59.41, 60.52, 60.33, 60.33],
            2: [57.01, 61.44, 62.55, 62.55, 62.73],
            3: [57.01, 61.44, 62.55, 62.55, 62.73],
        },
    },
    "pubmed": {
        False: {
            1: [74.77, 80.98, 81.68, 81.78, 82.08],
            2: [85.19, 88.89, 88.99, 88.89, 90.29],
            3: [85.19, 88.89, 88.99, 88.89, 90.29],
        },
        True: {
            1: [74.97, 76.78, 77.58, 77.88, 77.98],
            2: [77.08, 81.88, 82.78, 82.28, 82.68],
            3: [77.08, 81.88, 82.78, 82.28, 82.68],
        },
    },
}


def plot_nb_sweep() -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), dpi=220, sharex=True)
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for row, dataset in enumerate(["cora", "pubmed"]):
        for col, topo in enumerate([False, True]):
            ax = axes[row][col]
            for hop in HOP_VALUES:
                ax.plot(
                    NB_VALUES,
                    SWEEP[dataset][topo][hop],
                    marker="o",
                    linewidth=2.4,
                    markersize=4.8,
                    color=colors[hop],
                    label=f"{hop}-hop",
                )
            ax.set_title(f"{dataset.capitalize()} | topo={topo}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Neighbor Count", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT_DIR / "nb_sweep_cora_pubmed.png"
    pdf = OUT_DIR / "nb_sweep_cora_pubmed.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def plot_hop_sweep() -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), dpi=220, sharex=True)
    colors = {1: "#4C78A8", 3: "#72B7B2", 5: "#F58518", 10: "#E45756", 20: "#54A24B"}
    for row, dataset in enumerate(["cora", "pubmed"]):
        for col, topo in enumerate([False, True]):
            ax = axes[row][col]
            for nb in NB_VALUES:
                vals = [SWEEP[dataset][topo][hop][NB_VALUES.index(nb)] for hop in HOP_VALUES]
                ax.plot(
                    HOP_VALUES,
                    vals,
                    marker="o",
                    linewidth=2.2,
                    markersize=4.6,
                    color=colors[nb],
                    label=f"nb={nb}",
                )
            ax.set_title(f"{dataset.capitalize()} | topo={topo}", fontsize=18, fontweight="bold")
            ax.set_xlabel("Hop Count", fontsize=16)
            ax.set_ylabel("Accuracy (%)", fontsize=16)
            ax.set_xticks(HOP_VALUES)
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.tick_params(axis="both", labelsize=14)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT_DIR / "hop_sweep_cora_pubmed.png"
    pdf = OUT_DIR / "hop_sweep_cora_pubmed.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def main() -> None:
    setup_style()
    outputs = []
    outputs += plot_cora_first_last_neighbor_ablation()
    outputs += plot_pubmed_mask_sweep()
    outputs += plot_nb_sweep()
    outputs += plot_hop_sweep()
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
