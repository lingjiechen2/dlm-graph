"""
Plot all Cora checkpoint eval results (logit + infill) from a summary CSV.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/plot_cora_all_ckpts_eval.py \
        --summary_csv /home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_all_ckpts_eval_20260423_175543/summary.csv \
        --out_dir /home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_all_ckpts_eval_20260423_175543
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_step(checkpoint_name: str) -> int:
    if checkpoint_name == "checkpoint-final":
        return 10**9
    return int(checkpoint_name.replace("checkpoint-", ""))


def short_run_name(run_dir_name: str) -> str:
    name = run_dir_name.replace("tmdlm-llada-8b-cora-", "")
    name = name.replace("-base-lora-20260418_185914", "")
    name = name.replace("-base-lora", "")
    return name


def extract_metric(df: pd.DataFrame) -> pd.Series:
    if "accuracy" in df.columns and df["accuracy"].notna().any():
        return pd.to_numeric(df["accuracy"], errors="coerce")
    return pd.to_numeric(df["accuracy_lenient"], errors="coerce")


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
    pad = max(0.6, span * 0.12)
    return (lo - pad, hi + pad)


def make_run_panels(df: pd.DataFrame, out_path: Path) -> None:
    run_names = sorted(df["run_dir_name"].unique())
    n = len(run_names)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), dpi=220)
    axes = axes.flatten()

    color_logit = "#4C4C4C"
    color_infill = "#D55E00"

    for i, run in enumerate(run_names):
        ax = axes[i]
        sub = df[df["run_dir_name"] == run].copy()
        sub["step"] = sub["checkpoint_name"].map(parse_step)
        sub = sub.sort_values("step")

        s_logit = sub[sub["eval_type"] == "logit"].copy()
        s_logit["metric"] = pd.to_numeric(s_logit["accuracy"], errors="coerce")
        s_infill = sub[sub["eval_type"] == "infill"].copy()
        s_infill["metric"] = pd.to_numeric(s_infill["accuracy_lenient"], errors="coerce")

        if not s_logit.empty:
            ax.plot(
                s_logit["step"],
                s_logit["metric"],
                marker="o",
                linewidth=2.4,
                markersize=4.5,
                color=color_logit,
                label="Logit",
            )
        if not s_infill.empty:
            ax.plot(
                s_infill["step"],
                s_infill["metric"],
                marker="o",
                linewidth=2.4,
                markersize=4.5,
                color=color_infill,
                label="Infill (lenient)",
            )

        ax.set_title(short_run_name(run), fontsize=12, fontweight="bold")
        ax.set_xlabel("Checkpoint Step", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.25)
        ylim_vals = list(s_logit["metric"].dropna().values) + list(
            s_infill["metric"].dropna().values
        )
        y0, y1 = adaptive_ylim(ylim_vals, min_span=3.0)
        ax.set_ylim(y0, y1)
        ax.legend(frameon=False, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Cora: All Checkpoints (Logit vs Infill)", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_best_bar(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for run, sub in df.groupby("run_dir_name"):
        s_logit = sub[sub["eval_type"] == "logit"].copy()
        s_logit["metric"] = pd.to_numeric(s_logit["accuracy"], errors="coerce")
        s_infill = sub[sub["eval_type"] == "infill"].copy()
        s_infill["metric"] = pd.to_numeric(s_infill["accuracy_lenient"], errors="coerce")

        logit_best = s_logit["metric"].max() if not s_logit.empty else float("nan")
        infill_best = s_infill["metric"].max() if not s_infill.empty else float("nan")
        rows.append(
            {
                "run_dir_name": run,
                "run_label": short_run_name(run),
                "logit_best": logit_best,
                "infill_best": infill_best,
            }
        )
    best_df = pd.DataFrame(rows).sort_values("run_label")

    fig, ax = plt.subplots(figsize=(15, 5.6), dpi=220)
    x = range(len(best_df))
    w = 0.34
    ax.bar(
        [i - w / 2 for i in x],
        best_df["logit_best"],
        width=w,
        color="#6E6E6E",
        label="Best Logit",
    )
    ax.bar(
        [i + w / 2 for i in x],
        best_df["infill_best"],
        width=w,
        color="#D55E00",
        label="Best Infill (lenient)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(best_df["run_label"], rotation=22, ha="right")
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title("Cora: Best Checkpoint Accuracy per Run")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    yvals = list(best_df["logit_best"].dropna().values) + list(
        best_df["infill_best"].dropna().values
    )
    y0, y1 = adaptive_ylim(yvals, min_span=3.0)
    ax.set_ylim(y0, y1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_csv)
    df = df[df["return_code"] == 0].copy()

    panel_out = out_dir / "cora_all_ckpts_run_panels.png"
    bar_out = out_dir / "cora_all_ckpts_best_bar.png"

    make_run_panels(df, panel_out)
    make_best_bar(df, bar_out)

    print(f"Saved: {panel_out}")
    print(f"Saved: {bar_out}")


if __name__ == "__main__":
    main()
