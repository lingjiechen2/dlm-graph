"""
Line plot of ogbn-arxiv accuracy vs `max_neighbors_per_hop` (TTA neighbor-count
sweep) for the §21 r128 SFT, evaluated at N=5000 on top ckpts.

Visual style loaded from analysis/plot_style.json.

Data sources:
    - Phase 6 JSONL: phase6_5k_nb_sweep.jsonl (16 runs, 4 ckpts × 4 nb)
    - LLaGA reference numbers: hard-coded from arXiv:2402.08170 Table 1

Usage:
    python analysis/plot_arxiv_nb_sweep.py \
        --jsonl analysis/postprocess_arxiv_r128/eval_jsonl/phase6_5k_nb_sweep.jsonl \
        --out analysis/figures/arxiv_nb_sweep.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _plot_utils import apply_rcparams, hide_spines_from_style, load_style


# LLaGA Single Focus baselines (full test, ~48,604 samples)
LLAGA_BASELINES = {
    "LLaGA-HO-7B": 76.66,
    "LLaGA-ND-7B": 75.98,
}


def load_phase6(jsonl_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Returns {ckpt: [(nb, acc), ...]} from the phase6 JSONL."""
    by_ckpt: dict[str, dict[int, float]] = defaultdict(dict)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            exp = d.get("experiment", "")
            # phase6-<ckpt>_nb<NB>
            if "phase6-" not in exp:
                continue
            tag = exp.replace("phase6-", "")  # e.g., 1845_nb12
            try:
                ck, nb_part = tag.split("_nb")
                nb = int(nb_part)
            except ValueError:
                continue
            acc = d.get("accuracy")
            if acc is not None:
                by_ckpt[ck][nb] = float(acc)
    # sort each ckpt's points by nb
    return {ck: sorted(d.items()) for ck, d in by_ckpt.items()}


def plot_baselines(ax, baselines: dict[str, float], style: dict,
                   ylim_lo: float, ylim_hi: float) -> None:
    bs = style["baseline"]
    sorted_bl = sorted(baselines.items(), key=lambda x: -x[1])
    label_y: list[float] = []
    for (label, value), color in zip(sorted_bl, bs["colors"]):
        ax.axhline(value, color=color, linestyle=bs["linestyle"],
                   linewidth=bs["linewidth"], alpha=bs["alpha"], zorder=1)
        if ylim_lo <= value <= ylim_hi:
            y_text = value
            for prev in label_y:
                if abs(y_text - prev) < bs["min_label_gap"]:
                    y_text = prev - bs["min_label_gap"]
            label_y.append(y_text)
            ax.annotate(f"{label} ({value:.2f})",
                        xy=(1.0, y_text), xycoords=("axes fraction", "data"),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=bs["label_fontsize"], color=color,
                        va="bottom", clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="analysis/postprocess_arxiv_r128/eval_jsonl/phase6_5k_nb_sweep.jsonl",
    )
    parser.add_argument("--style", default="analysis/plot_style.json")
    parser.add_argument("--out", default="analysis/figures/arxiv_nb_sweep.png")
    parser.add_argument("--skip_final", action=argparse.BooleanOptionalAction, default=True,
                        help="ckpt-final == ckpt-2042 by md5 — skip to avoid clutter")
    args = parser.parse_args()

    style = load_style(Path(args.style))
    apply_rcparams(style)
    hide_spines_from_style(style)

    data = load_phase6(Path(args.jsonl))
    if not data:
        raise SystemExit(f"No data found in {args.jsonl}")
    if args.skip_final and "final" in data:
        del data["final"]

    # Per-ckpt visual styling — use a small palette for 3 unique ckpts
    # 1640 = early/weakest, 1845 = best, 2042 = late (same as final)
    ckpt_style = {
        "1640":  {"color": "#92C5DE", "marker": "o", "linestyle": "-",  "label": r"\texttt{ckpt-1640}"},
        "1845":  {"color": "#2166AC", "marker": "s", "linestyle": "-",  "label": r"\texttt{ckpt-1845} (best)"},
        "2042":  {"color": "#D55E00", "marker": "^", "linestyle": "--", "label": r"\texttt{ckpt-2042} = \texttt{ckpt-final}"},
    }

    s = style["series"]
    fig, ax = plt.subplots(figsize=style["figure"]["figsize"],
                           dpi=style["figure"]["dpi"])

    all_acc = []
    # Sort by ckpt step for legend order
    ordered = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 99999)
    for ck in ordered:
        pts = data[ck]
        if not pts:
            continue
        nbs, accs = zip(*pts)
        sp = ckpt_style.get(ck, {"color": "#888", "marker": "x", "linestyle": ":", "label": ck})
        ax.plot(nbs, accs, color=sp["color"], linestyle=sp["linestyle"],
                marker=sp["marker"], linewidth=s["linewidth"],
                markersize=s["markersize"], label=sp["label"])
        # Endpoint label on the right-most point
        ax.annotate(f"{accs[-1]:.2f}",
                    xy=(nbs[-1], accs[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=s["endpoint_label_fontsize"],
                    color=sp["color"], va="center")
        all_acc.extend(accs)

    # σ at N=5000 ≈ sqrt(0.75 * 0.25 / 5000) ≈ 0.6 pt
    sigma = 0.61

    lo_sft, hi_sft = min(all_acc), max(all_acc)
    # Include LLaGA baselines in y-range padding
    extreme = list(all_acc) + list(LLAGA_BASELINES.values())
    pad = max(0.6, (max(extreme) - min(extreme)) * 0.12)
    ylim_lo = min(extreme) - pad
    ylim_hi = max(extreme) + pad
    ax.set_ylim(ylim_lo, ylim_hi)

    # SFT-default nb=10 reference vertical line (label rotated alongside line)
    ax.axvline(10, color="#444", linestyle=":", linewidth=1.0, alpha=0.7, zorder=0)
    y_mid = ylim_lo + (ylim_hi - ylim_lo) * 0.55
    ax.annotate("SFT-default",
                xy=(10, y_mid), xytext=(-10, 0),
                textcoords="offset points",
                fontsize=7.5, color="#444", va="center", ha="right",
                rotation=90)

    plot_baselines(ax, LLAGA_BASELINES, style, ylim_lo, ylim_hi)

    # Annotate sample-size / noise floor in bottom-left
    ax.text(0.02, 0.04,
            r"$N{=}5000$, $\sigma{\approx}" + f"{sigma:.2f}" + r"$\,pt",
            transform=ax.transAxes, fontsize=8, va="bottom",
            color="#444")

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel(r"\texttt{max\_neighbors\_per\_hop}",
                  fontsize=style["axes"]["label_fontsize"])
    ax.set_ylabel(r"Accuracy (\%)", fontsize=style["axes"]["label_fontsize"])
    ax.set_title("ogbn-arxiv: TTA neighbor-count sweep",
                 fontsize=style["title"]["fontsize"],
                 pad=style["title"]["pad"])

    # Add ticks at each tested nb
    tested_nbs = sorted({nb for pts in data.values() for nb, _ in pts})
    ax.set_xticks(tested_nbs)

    lg = style["legend"]
    ax.legend(frameon=lg["frameon"], fontsize=lg["fontsize"], loc=lg["loc"])
    g = style["grid"]
    ax.grid(axis=g["axis"], linestyle=g["linestyle"], alpha=g["alpha"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
