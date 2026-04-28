"""Run PubMed final-checkpoint eval_infill with a 5-token generation window.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    CUDA_VISIBLE_DEVICES=0 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_final_infill_mask5.py

Examples:
    CUDA_VISIBLE_DEVICES=0 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_final_infill_mask5.py
    CUDA_VISIBLE_DEVICES=0 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_final_infill_mask5.py --topo_tag notopo
    CUDA_VISIBLE_DEVICES=0 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_final_infill_mask5.py --topo_tag topo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
PYTHON = "/home/lingjie7/anaconda3/envs/dllm/bin/python"
MODEL = "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
SUMMARY_DIR = REPO_ROOT / "summaries" / "pubmed_final_infill_mask5_compare"

CHECKPOINTS = {
    "notopo": REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-pubmed-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-20260425_150500"
    / "checkpoint-final",
    "topo": REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-pubmed-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-20260425_150500"
    / "checkpoint-final",
}


def build_cmd(topo_tag: str) -> list[str]:
    use_topology_mask = "True" if topo_tag == "topo" else "False"
    exp_name = f"pubmed_final_infill_{topo_tag}_mask5"
    log_file = SUMMARY_DIR / f"{exp_name}.jsonl"
    stdout_file = SUMMARY_DIR / f"{exp_name}.out"
    cmd = [
        PYTHON,
        str(REPO_ROOT / "examples" / "tmdlm" / "eval_infill.py"),
        "--exp",
        exp_name,
        "--model_name_or_path",
        MODEL,
        "--lora_path",
        str(CHECKPOINTS[topo_tag]),
        "--dataset_name",
        "pubmed",
        "--split",
        "test",
        "--batch_size",
        "1",
        "--max_seq_len",
        "2048",
        "--max_neighbors_per_hop",
        "10",
        "--max_hops",
        "2",
        "--steps",
        "10",
        "--temperature",
        "0.0",
        "--remasking",
        "low_confidence",
        "--prompt_format",
        "category_infill",
        "--prompt_layout",
        "target_first",
        "--max_answer_tokens",
        "6",
        "--max_new_tokens",
        "5",
        "--include_neighbor_labels",
        "True",
        "--neighbor_label_format",
        "bracket",
        "--use_topology_mask",
        use_topology_mask,
        "--seed",
        "42",
        "--log_file",
        str(log_file),
    ]
    return cmd, stdout_file


def run_one(topo_tag: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    cmd, stdout_file = build_cmd(topo_tag)
    print(f"[run] {topo_tag}")
    print(" ".join(cmd))
    with stdout_file.open("w") as fh:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    print(f"[done] stdout={stdout_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topo_tag",
        default="both",
        choices=["both", "notopo", "topo"],
        help="Which final checkpoint(s) to evaluate.",
    )
    args = parser.parse_args()

    topo_tags = ["notopo", "topo"] if args.topo_tag == "both" else [args.topo_tag]
    for topo_tag in topo_tags:
        run_one(topo_tag)


if __name__ == "__main__":
    main()
