"""Show real PubMed infill outputs with a 6-token generation window.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    CUDA_VISIBLE_DEVICES=1 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/show_pubmed_infill_mask6_outputs.py

Examples:
    CUDA_VISIBLE_DEVICES=1 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/show_pubmed_infill_mask6_outputs.py
    CUDA_VISIBLE_DEVICES=1 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/show_pubmed_infill_mask6_outputs.py --topo
    CUDA_VISIBLE_DEVICES=1 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/show_pubmed_infill_mask6_outputs.py --max_samples 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
PYTHON = "/home/lingjie7/anaconda3/envs/dllm/bin/python"
MODEL = "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
NOTOPO_FINAL = (
    REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-pubmed-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-20260425_150500"
    / "checkpoint-final"
)
TOPO_FINAL = (
    REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-pubmed-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-20260425_150500"
    / "checkpoint-final"
)


def build_cmd(args: argparse.Namespace) -> list[str]:
    lora_path = TOPO_FINAL if args.topo else NOTOPO_FINAL
    use_topology_mask = "True" if args.topo else "False"
    exp_name = "pubmed_infill_mask6_topo_outputs" if args.topo else "pubmed_infill_mask6_notopo_outputs"
    log_file = (
        REPO_ROOT
        / "summaries"
        / "pubmed_infill_mask6_outputs"
        / f"{exp_name}.jsonl"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    return [
        PYTHON,
        str(REPO_ROOT / "examples" / "tmdlm" / "eval_infill.py"),
        "--exp",
        exp_name,
        "--model_name_or_path",
        MODEL,
        "--lora_path",
        str(lora_path),
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
        "6",
        "--include_neighbor_labels",
        "True",
        "--neighbor_label_format",
        "bracket",
        "--use_topology_mask",
        use_topology_mask,
        "--max_samples",
        str(args.max_samples),
        "--log_file",
        str(log_file),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topo",
        action="store_true",
        help="Use the topo final checkpoint instead of notopo.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="How many samples to run. eval_infill.py will still print the first 10 decoded answers.",
    )
    args = parser.parse_args()

    cmd = build_cmd(args)
    print("Running command:")
    print(" ".join(cmd))
    print()
    subprocess.run(cmd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
