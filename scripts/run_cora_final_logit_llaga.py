"""Run Cora final-checkpoint eval_logit on the LLaGA-aligned dataset.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    CUDA_VISIBLE_DEVICES=7 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_logit_llaga.py

Examples:
    CUDA_VISIBLE_DEVICES=7 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_logit_llaga.py
    CUDA_VISIBLE_DEVICES=7 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_logit_llaga.py --topo_tag notopo
    CUDA_VISIBLE_DEVICES=7 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_logit_llaga.py --topo_tag topo
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
PYTHON = "/home/lingjie7/anaconda3/envs/dllm/bin/python"
MODEL = "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
SUMMARY_DIR = REPO_ROOT / "summaries" / "cora_final_logit_llaga"

CHECKPOINTS = {
    "notopo": REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-cora-2hop-notopo-catinfill-nbmask-noeospad-r64-ep20-20260425_042600"
    / "checkpoint-final",
    "topo": REPO_ROOT
    / ".models"
    / "tmdlm-llada-8b-cora-2hop-topo-catinfill-nbmask-noeospad-r64-ep20-20260425_042600"
    / "checkpoint-final",
}


def build_cmd(topo_tag: str) -> tuple[list[str], Path]:
    use_topology_mask = "True" if topo_tag == "topo" else "False"
    exp_name = f"cora_final_logit_llaga_{topo_tag}"
    log_file = SUMMARY_DIR / f"{exp_name}.jsonl"
    stdout_file = SUMMARY_DIR / f"{exp_name}.out"
    cmd = [
        PYTHON,
        str(REPO_ROOT / "examples" / "tmdlm" / "eval_logit.py"),
        "--exp",
        exp_name,
        "--model_name_or_path",
        MODEL,
        "--lora_path",
        str(CHECKPOINTS[topo_tag]),
        "--dataset_name",
        "cora",
        "--split",
        "test",
        "--batch_size",
        "8",
        "--max_seq_len",
        "2048",
        "--max_neighbors_per_hop",
        "10",
        "--max_hops",
        "2",
        "--prompt_format",
        "category_infill",
        "--prompt_layout",
        "target_first",
        "--max_answer_tokens",
        "6",
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
