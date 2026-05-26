#!/usr/bin/env python3
"""Update markdown with frozen LLaDA LP results from eval JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "<!-- frozen-llada-lp-start -->"
END = "<!-- frozen-llada-lp-end -->"
DATASET_ORDER = ["cora", "pubmed", "ogbn-arxiv"]
DISPLAY = {"cora": "Cora", "pubmed": "PubMed", "ogbn-arxiv": "ogbn-arxiv"}


def _latest_entries(log_file: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not log_file.exists():
        return latest
    with log_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task") not in (None, "lp"):
                continue
            if obj.get("split") != "llaga_test":
                continue
            cfg = obj.get("config", {})
            if cfg.get("lora_path") not in (None, "", "None"):
                continue
            dataset = obj.get("dataset")
            if dataset in DATASET_ORDER:
                latest[dataset] = obj
    return latest


def _fmt(value, ndigits: int = 2) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{ndigits}f}"


def _render(log_file: Path, entries: dict[str, dict]) -> str:
    lines = [
        START,
        "## Frozen LLaDA-8B-Instruct LP Eval (LLaGA Splits)",
        "",
        "Zero-shot baseline using the untrained/non-SFT `GSAI-ML/LLaDA-8B-Instruct` model on the official LLaGA LP test splits. The prompt/eval setup matches the LP head-to-head eval: `max_seq_len=4096`, 2-hop neighborhoods, 10 neighbors per hop, sequential positions, and topology mask enabled.",
        "",
        "| Dataset | Samples | Accuracy | AUC | Per-label acc (no / yes) |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in DATASET_ORDER:
        obj = entries.get(dataset)
        if obj is None:
            lines.append(f"| {DISPLAY[dataset]} | pending | pending | pending | pending |")
            continue
        per = obj.get("per_label_accuracy", {})
        lines.append(
            "| {dataset} | {n} | {acc} | {auc} | {no} / {yes} |".format(
                dataset=DISPLAY[dataset],
                n=obj.get("n_samples", "pending"),
                acc=_fmt(obj.get("accuracy")),
                auc=_fmt(obj.get("auc"), 4),
                no=_fmt(per.get("no")),
                yes=_fmt(per.get("yes")),
            )
        )
    lines.extend(
        [
            "",
            f"JSONL: `{log_file}`",
            "",
            END,
        ]
    )
    return "\n".join(lines)


def _replace_section(path: Path, section: str) -> None:
    if path.exists():
        text = path.read_text()
    else:
        text = ""
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        new_text = f"{before}\n\n{section}\n\n{after}".rstrip() + "\n"
    else:
        new_text = text.rstrip() + "\n\n" + section + "\n"
    path.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--results-md", default="results.md", type=Path)
    parser.add_argument(
        "--detailed-md", default="results/current_results_detailed.md", type=Path
    )
    args = parser.parse_args()

    entries = _latest_entries(args.log_file)
    section = _render(args.log_file, entries)
    _replace_section(args.results_md, section)
    _replace_section(args.detailed_md, section)


if __name__ == "__main__":
    main()
