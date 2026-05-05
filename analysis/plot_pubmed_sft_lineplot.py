"""
Line plot of PubMed SFT checkpoint accuracy (logit + infill, topo + notopo).
Visual style loaded from analysis/plot_style.json.

Usage:
    python analysis/plot_pubmed_sft_lineplot.py \
        --jsonl_dir /tmp/dlm-graph-eval-jsonl \
        --run_tag pubmed_20260428_aligned \
        --out analysis/figures/pubmed_sft_lineplot.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_series(jsonl_path: Path) -> list[tuple[int, float]]:
    points = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            lora_path = d.get("config", {}).get("lora_path", "")
            if "checkpoint-" not in lora_path:
                continue
            step = int(lora_path.split("checkpoint-")[-1])
            acc = d.get("accuracy", d.get("accuracy_strict"))
            if acc is not None:
                points.append((step, float(acc)))
    return sorted(points)


def load_baselines(json_path: Path) -> dict[str, float]:
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    return {k: float(v) for k, v in raw.items() if not k.startswith("_")}


def load_style(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def apply_rcparams(style: dict) -> None:
    f = style["font"]
    plt.rcParams.update({
        "text.usetex": f.get("usetex", False),
        "text.latex.preamble": f.get("latex_preamble", ""),
        "font.family": f["family"],
        "axes.spines.top": style["axes"]["spines_top"],
        "axes.spines.right": style["axes"]["spines_right"],
        "xtick.labelsize": style["axes"]["tick_labelsize"],
        "ytick.labelsize": style["axes"]["tick_labelsize"],
    })


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
            ax.annotate(f"{label} ({value:.1f})",
                        xy=(1.0, y_text), xycoords=("axes fraction", "data"),
                        xytext=(4, 2), textcoords="offset points",
                        fontsize=bs["label_fontsize"], color=color,
                        va="bottom", clip_on=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", default="/tmp/dlm-graph-eval-jsonl")
    parser.add_argument("--run_tag", default="pubmed_20260502_mcdigit_nonb_seq4k")
    parser.add_argument("--baselines", default="analysis/baselines_pubmed.json")
    parser.add_argument("--style", default="analysis/plot_style.json")
    parser.add_argument("--out", default="analysis/figures/pubmed_sft_lineplot.png")
    args = parser.parse_args()

    base = Path(args.jsonl_dir)
    tag = args.run_tag
    baselines = load_baselines(Path(args.baselines))
    style = load_style(Path(args.style))
    apply_rcparams(style)

    # Logit-only, 2 lines (notopo + topo). Filename pattern auto-detected
    # from run_tag: seq4k uses the eval_logit-style filenames, seq2k uses
    # the older 2hop-* filenames.
    if "seq4k" in tag:
        series_paths = {
            ("logit", "notopo"): base / f"eval-pubmed-seq4k-notopo-mcdigit-{tag}-logit.jsonl",
            ("logit", "topo"):   base / f"eval-pubmed-seq4k-topo-mcdigit-{tag}-logit.jsonl",
        }
    else:
        series_paths = {
            ("logit", "notopo"): base / f"eval-pubmed-2hop-notopo-{tag}.jsonl",
            ("logit", "topo"):   base / f"eval-pubmed-2hop-topo-{tag}.jsonl",
        }
    palette_key = {
        ("logit", "notopo"): "logit_notopo",
        ("logit", "topo"):   "logit_topo",
    }

    s = style["series"]
    palette = s["palette"]

    fig, ax = plt.subplots(figsize=style["figure"]["figsize"],
                           dpi=style["figure"]["dpi"])

    all_acc = []
    for key, path in series_paths.items():
        if not path.exists():
            print(f"  missing: {path}")
            continue
        pts = load_series(path)
        if not pts:
            continue
        steps, accs = zip(*pts)
        p = palette[palette_key[key]]
        ax.plot(steps, accs, color=p["color"], linestyle=p["linestyle"],
                marker=p["marker"], linewidth=s["linewidth"],
                markersize=s["markersize"], label=p["label"])
        ax.annotate(f"{accs[-1]:.1f}",
                    xy=(steps[-1], accs[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=s["endpoint_label_fontsize"],
                    color=p["color"], va="center")
        all_acc.extend(accs)

    lo_sft, hi_sft = min(all_acc), max(all_acc)
    pad = max(0.3, (hi_sft - lo_sft) * 0.15)
    ylim_lo, ylim_hi = lo_sft - pad, hi_sft + pad
    ax.set_ylim(ylim_lo, ylim_hi)

    plot_baselines(ax, baselines, style, ylim_lo, ylim_hi)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Checkpoint Step", fontsize=style["axes"]["label_fontsize"])
    ax.set_ylabel(r"Accuracy (\%)", fontsize=style["axes"]["label_fontsize"])
    ax.set_title("PubMed", fontsize=style["title"]["fontsize"],
                 pad=style["title"]["pad"])
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
