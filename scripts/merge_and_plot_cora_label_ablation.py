"""
Merge Cora eval results (neighbor-label off + on) and plot full results.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/merge_and_plot_cora_label_ablation.py \
      --label_off_csv /home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_all_ckpts_eval_20260423_175543/summary.csv \
      --label_on_csv /home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_first_last_nb_labels_eval_20260423_200547/summary.csv \
      --out_dir /home/lingjie7/auto-research/projects/dlm-graph/summaries/cora_label_ablation_merged
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
    return name


def adaptive_ylim(values: list[float], min_span: float = 2.0) -> tuple[float, float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span < min_span:
        center = (lo + hi) / 2.0
        half = min_span / 2.0
        return (center - half, center + half)
    pad = max(0.6, span * 0.12)
    return (lo - pad, hi + pad)


def build_long_df(label_off_csv: Path, label_on_csv: Path) -> pd.DataFrame:
    off = pd.read_csv(label_off_csv).copy()
    on = pd.read_csv(label_on_csv).copy()

    off["include_neighbor_labels"] = False
    if "include_neighbor_labels" not in on.columns:
        on["include_neighbor_labels"] = True

    merged = pd.concat([off, on], ignore_index=True, sort=False)
    merged = merged[merged["return_code"] == 0].copy()

    # Unified metric column for plotting:
    # - logit: accuracy
    # - infill: accuracy_lenient
    merged["metric"] = pd.NA
    is_logit = merged["eval_type"] == "logit"
    is_infill = merged["eval_type"] == "infill"
    merged.loc[is_logit, "metric"] = pd.to_numeric(
        merged.loc[is_logit, "accuracy"], errors="coerce"
    )
    merged.loc[is_infill, "metric"] = pd.to_numeric(
        merged.loc[is_infill, "accuracy_lenient"], errors="coerce"
    )

    merged["step"] = merged["checkpoint_name"].map(parse_step)
    merged["run_label"] = merged["run_dir_name"].map(short_run_name)
    return merged


def plot_full_panels(df: pd.DataFrame, out_path: Path) -> None:
    runs = sorted(df["run_dir_name"].unique())
    n = len(runs)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), dpi=220)
    axes = axes.flatten()

    color_logit_off = "#4C4C4C"
    color_infill_off = "#D55E00"
    color_logit_on = "#1F77B4"
    color_infill_on = "#2CA02C"

    for i, run in enumerate(runs):
        ax = axes[i]
        sub = df[df["run_dir_name"] == run].copy()

        def _sel(eval_type: str, with_labels: bool) -> pd.DataFrame:
            d = sub[
                (sub["eval_type"] == eval_type)
                & (sub["include_neighbor_labels"] == with_labels)
            ].copy()
            return d.sort_values("step")

        logit_off = _sel("logit", False)
        infill_off = _sel("infill", False)
        logit_on = _sel("logit", True)
        infill_on = _sel("infill", True)

        if not logit_off.empty:
            ax.plot(
                logit_off["step"],
                logit_off["metric"],
                color=color_logit_off,
                linewidth=2.2,
                marker="o",
                markersize=4.2,
                label="Logit (label off)",
            )
        if not infill_off.empty:
            ax.plot(
                infill_off["step"],
                infill_off["metric"],
                color=color_infill_off,
                linewidth=2.2,
                marker="o",
                markersize=4.2,
                label="Infill (label off)",
            )

        # label-on only has first/last checkpoints; highlight as dashed markers
        if not logit_on.empty:
            ax.plot(
                logit_on["step"],
                logit_on["metric"],
                color=color_logit_on,
                linewidth=1.9,
                linestyle="--",
                marker="D",
                markersize=5.0,
                label="Logit (label on)",
            )
        if not infill_on.empty:
            ax.plot(
                infill_on["step"],
                infill_on["metric"],
                color=color_infill_on,
                linewidth=1.9,
                linestyle="--",
                marker="D",
                markersize=5.0,
                label="Infill (label on)",
            )

        y_vals = (
            list(logit_off["metric"].dropna().values)
            + list(infill_off["metric"].dropna().values)
            + list(logit_on["metric"].dropna().values)
            + list(infill_on["metric"].dropna().values)
        )
        y0, y1 = adaptive_ylim(y_vals, min_span=3.0)
        ax.set_ylim(y0, y1)

        ax.set_title(short_run_name(run), fontsize=12, fontweight="bold")
        ax.set_xlabel("Checkpoint Step", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Cora Full Results: Neighbor Label Off vs On",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_off_csv", required=True)
    parser.add_argument("--label_on_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif", "Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = build_long_df(Path(args.label_off_csv), Path(args.label_on_csv))
    merged_csv = out_dir / "merged_results.csv"
    merged_json = out_dir / "merged_results.json"
    merged.to_csv(merged_csv, index=False)
    merged.to_json(merged_json, orient="records", indent=2)

    panel_png = out_dir / "cora_label_ablation_full_panels.png"
    plot_full_panels(merged, panel_png)

    print(f"Saved: {merged_csv}")
    print(f"Saved: {merged_json}")
    print(f"Saved: {panel_png}")


if __name__ == "__main__":
    main()

