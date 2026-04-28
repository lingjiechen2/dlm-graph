"""Plot PubMed all-checkpoint SFT evaluation results.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/plot_pubmed_all_ckpts_eval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_SUMMARY_CSV = Path(
    "/home/lingjie7/auto-research/projects/dlm-graph/summaries/"
    "pubmed_allckpts_eval_gpu0126_even_20260425_235521/summary.csv"
)
DEFAULT_OUT_DIR = DEFAULT_SUMMARY_CSV.parent

BASELINES = {
    "SAGN": 95.17,
    "LLaGA-ND-7B": 95.03,
    "LLaGA-HO-7B": 95.03,
    "NodeFormer": 94.90,
    "GraphSAGE": 94.87,
    "GCN": 92.96,
    "GAT": 92.33,
    "SGC": 87.35,
}


def checkpoint_to_epoch(checkpoint_name: str, max_step: int) -> float:
    if checkpoint_name == "checkpoint-final":
        return 20.0
    step = int(checkpoint_name.replace("checkpoint-", ""))
    return step * 20.0 / max_step


def adaptive_ylim(values: list[float], min_span: float = 2.0) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span < min_span:
        center = (hi + lo) / 2.0
        half = min_span / 2.0
        return (center - half, center + half)
    pad = max(0.4, span * 0.12)
    return (lo - pad, hi + pad)


def short_run_name(run_name: str) -> str:
    return "Topo" if "topo" in run_name and "notopo" not in run_name else "No-Topo"


def prepare_eval_df(df: pd.DataFrame, eval_type: str, max_step: int) -> pd.DataFrame:
    sub = df[df["eval_type"] == eval_type].copy()
    metric_col = "accuracy" if eval_type == "logit" else "accuracy_lenient"
    sub["metric"] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub[sub["metric"].notna()].copy()
    sub["epoch"] = sub["checkpoint_name"].map(lambda x: checkpoint_to_epoch(x, max_step))
    sub["is_final"] = sub["checkpoint_name"].eq("checkpoint-final").astype(int)
    sub = sub.sort_values(["epoch", "is_final"])
    sub = sub.drop_duplicates(subset=["epoch"], keep="last")
    return sub


def make_lines(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=220, sharey=True)
    color_logit = "#4C4C4C"
    color_infill = "#D55E00"
    baseline_palette = [
        "#9A9A9A",
        "#A7A7A7",
        "#B4B4B4",
        "#C1C1C1",
        "#CECECE",
        "#BBBBBB",
        "#D8D8D8",
        "#E1E1E1",
    ]

    nonfinal_steps = []
    for name in df["checkpoint_name"].dropna().unique():
        if name != "checkpoint-final":
            nonfinal_steps.append(int(name.replace("checkpoint-", "")))
    max_step = max(nonfinal_steps)

    for ax, topo in zip(axes, [False, True]):
        sub = df[df["use_topology_mask"] == topo].copy()
        logit_df = prepare_eval_df(sub, "logit", max_step)
        infill_df = prepare_eval_df(sub, "infill", max_step)

        ax.plot(
            logit_df["epoch"],
            logit_df["metric"],
            marker="o",
            linewidth=2.4,
            markersize=4.6,
            color=color_logit,
            label="Logit",
        )
        ax.plot(
            infill_df["epoch"],
            infill_df["metric"],
            marker="o",
            linewidth=2.4,
            markersize=4.6,
            color=color_infill,
            label="Infill",
        )

        for (label, value), baseline_color in zip(BASELINES.items(), baseline_palette):
            ax.axhline(
                value,
                color=baseline_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.95,
                label=label,
                zorder=1,
            )

        y_vals = list(logit_df["metric"].dropna().values) + list(infill_df["metric"].dropna().values)
        y0, y1 = adaptive_ylim(y_vals, min_span=3.0)
        y0 = min(y0, min(BASELINES.values()) - 0.4)
        y1 = max(y1, max(BASELINES.values()) + 0.4)
        ax.set_ylim(y0, y1)
        ax.set_title(f"PubMed {short_run_name(sub['run_name'].iloc[0])}", fontsize=18, fontweight="bold", pad=6)
        ax.set_xlabel("Training Epoch", fontsize=18)
        ax.set_xticks(list(range(2, 21, 2)))
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=28, labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

    axes[0].set_ylabel("Accuracy (%)", fontsize=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        fontsize=13.5,
        columnspacing=1.0,
        handletextpad=0.7,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_best_bar(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for topo in [False, True]:
        sub = df[df["use_topology_mask"] == topo].copy()
        logit_best = pd.to_numeric(sub[sub["eval_type"] == "logit"]["accuracy"], errors="coerce").max()
        infill_best = pd.to_numeric(sub[sub["eval_type"] == "infill"]["accuracy_lenient"], errors="coerce").max()
        rows.append({"setting": "No-Topo" if not topo else "Topo", "logit": logit_best, "infill": infill_best})

    best_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=220)
    x = range(len(best_df))
    width = 0.32
    ax.bar([i - width / 2 for i in x], best_df["logit"], width=width, color="#6E6E6E", label="Best Logit")
    ax.bar([i + width / 2 for i in x], best_df["infill"], width=width, color="#D55E00", label="Best Infill")
    ax.set_xticks(list(x))
    ax.set_xticklabels(best_df["setting"], fontsize=13)
    ax.set_ylabel("Best Accuracy (%)", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, fontsize=12)
    y_vals = list(best_df["logit"].dropna().values) + list(best_df["infill"].dropna().values)
    y0, y1 = adaptive_ylim(y_vals, min_span=2.5)
    ax.set_ylim(y0, y1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df = df[df["return_code"] == 0].copy()

    line_out = out_dir / "pubmed_all_ckpts_lines.png"
    bar_out = out_dir / "pubmed_all_ckpts_best_bar.png"
    make_lines(df, line_out)
    make_best_bar(df, bar_out)
    print(f"Saved: {line_out}")
    print(f"Saved: {bar_out}")


if __name__ == "__main__":
    main()
