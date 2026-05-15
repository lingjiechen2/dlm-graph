"""
Cora SFT checkpoint accuracy: LoRA r=64 vs r=128 (same prompt format, no
neighbor labels). Pulls Cora eval JSONLs and plots one line per (r, setting).

Output: analysis/figures/cora_lora_rank.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from _plot_utils import apply_rcparams, hide_spines_from_style, load_style


# (LoRA rank, sequence length, topology) → JSONL basename inside --jsonl_dir
SERIES_FILES = {
    ("r64",  "seq2k", "topo"): "eval-cora-2hop-topo-nonb-cora_20260429_mcdigit_nonb_fixed.jsonl",
    ("r128", "seq2k", "topo"): "eval-cora-seq2k-topo-mcdigit-r128-cora_20260504-logit.jsonl",
}


def load_series(jsonl_path: Path) -> list[tuple[int, float]]:
    if not jsonl_path.exists():
        return []
    pts = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            lp = d.get("config", {}).get("lora_path", "") or ""
            if "checkpoint-" not in lp:
                continue
            step = int(lp.split("checkpoint-")[-1].rstrip("/"))
            a = d.get("accuracy")
            if a is None:
                continue
            pts.append((step, float(a)))
    return sorted(pts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl_dir", default="/tmp/dlm-graph-eval-jsonl")
    p.add_argument("--style", default="analysis/plot_style.json")
    p.add_argument("--out", default="analysis/figures/cora_lora_rank.png")
    args = p.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    series = {key: jsonl_dir / fname for key, fname in SERIES_FILES.items()}

    style = load_style(Path(args.style))
    apply_rcparams(style)
    hide_spines_from_style(style)

    s = style["series"]
    # Custom palette: rank determines color; topology determines linestyle
    palette = {
        ("r64",  "seq2k", "topo"):   {"color": "#2166AC", "ls": "-",  "m": "o",
                                       "label": r"LoRA $r{=}64$"},
        ("r128", "seq2k", "topo"):   {"color": "#D55E00", "ls": "-",  "m": "D",
                                       "label": r"LoRA $r{=}128$"},
    }

    fig, ax = plt.subplots(figsize=style["figure"]["figsize"],
                           dpi=style["figure"]["dpi"])

    all_acc = []
    for key, path in series.items():
        pts = load_series(path)
        if not pts:
            print(f"  missing/empty: {path}")
            continue
        steps, accs = zip(*pts)
        cfg = palette[key]
        ax.plot(steps, accs, color=cfg["color"], linestyle=cfg["ls"],
                marker=cfg["m"], linewidth=s["linewidth"],
                markersize=s["markersize"], label=cfg["label"])
        peak_a = max(accs)
        peak_step = steps[accs.index(peak_a)]
        ax.annotate(f"{peak_a:.1f}",
                    xy=(peak_step, peak_a),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=s["endpoint_label_fontsize"],
                    color=cfg["color"], va="bottom")
        all_acc.extend(accs)

    if not all_acc:
        raise SystemExit("No data found.")

    lo, hi = min(all_acc), max(all_acc)
    pad = max(0.5, (hi - lo) * 0.12)
    ax.set_ylim(lo - pad, hi + pad)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_xlabel("Checkpoint Step", fontsize=style["axes"]["label_fontsize"])
    ax.set_ylabel(r"Accuracy (\%)", fontsize=style["axes"]["label_fontsize"])
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
