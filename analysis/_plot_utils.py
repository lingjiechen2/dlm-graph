"""Shared helpers for analysis/plot_*.py scripts.

Keeps font/style/JSONL boilerplate in one place. Spine visibility is left to
callers — heatmap scripts want the default frame; lineplot scripts typically
hide top/right spines themselves.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_style(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def apply_rcparams(style: dict) -> None:
    f = style["font"]
    plt.rcParams.update({
        "text.usetex": f.get("usetex", False),
        "text.latex.preamble": f.get("latex_preamble", ""),
        "font.family": f["family"],
        "xtick.labelsize": style["axes"]["tick_labelsize"],
        "ytick.labelsize": style["axes"]["tick_labelsize"],
    })


def hide_spines_from_style(style: dict) -> None:
    """Apply spines_top/spines_right toggles when the style file defines them."""
    axes = style.get("axes", {})
    rc = {}
    if "spines_top" in axes:
        rc["axes.spines.top"] = axes["spines_top"]
    if "spines_right" in axes:
        rc["axes.spines.right"] = axes["spines_right"]
    if rc:
        plt.rcParams.update(rc)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
