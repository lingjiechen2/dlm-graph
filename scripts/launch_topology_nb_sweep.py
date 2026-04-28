"""
Launch topology-mask neighbor-count sweeps for TAG datasets.

Run:
    source ~/.zshrc
    conda activate dllm
    cd /home/lingjie7/auto-research/projects/dlm-graph
    python scripts/launch_topology_nb_sweep.py --max-parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
EVAL_SCRIPT = REPO_ROOT / "examples" / "tmdlm" / "eval_infill.py"
DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"


@dataclass(frozen=True)
class DatasetCfg:
    steps: int
    max_answer_tokens: int
    max_new_tokens: int
    alias: str


DATASET_CFGS: dict[str, DatasetCfg] = {
    "cora": DatasetCfg(steps=8, max_answer_tokens=4, max_new_tokens=8, alias="cora"),
    "pubmed": DatasetCfg(
        steps=10, max_answer_tokens=6, max_new_tokens=10, alias="pubmed"
    ),
    "ogbn-arxiv": DatasetCfg(
        steps=16, max_answer_tokens=8, max_new_tokens=16, alias="arxiv"
    ),
    "ogbn-products": DatasetCfg(
        steps=12, max_answer_tokens=6, max_new_tokens=12, alias="products"
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topology-mask neighbor sweep launcher")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--min-free-mem-mb", type=int, default=60000)
    parser.add_argument("--max-util-pct", type=int, default=20)
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def _detect_eligible_gpus(min_free_mem_mb: int, max_util_pct: int) -> list[int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    eligible: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        idx_s, free_s, util_s = [x.strip() for x in line.split(",")]
        idx = int(idx_s)
        free_mb = int(free_s)
        util_pct = int(util_s)
        fallback.append((idx, free_mb))
        if free_mb >= min_free_mem_mb and util_pct <= max_util_pct:
            eligible.append((idx, free_mb))

    # Prefer eligible GPUs by free memory desc; otherwise fallback to best free-memory GPUs.
    if eligible:
        eligible.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in eligible]
    fallback.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in fallback]


def _build_tasks(model_name_or_path: str) -> list[dict]:
    tasks: list[dict] = []
    for dataset, dcfg in DATASET_CFGS.items():
        for hops in (1, 2, 3):
            for nb in (1, 3, 5, 10, 20):
                exp = f"openended_{dcfg.alias}_{hops}hop_nb{nb}_topo"
                tasks.append(
                    {
                        "dataset": dataset,
                        "max_hops": hops,
                        "max_neighbors_per_hop": nb,
                        "steps": dcfg.steps,
                        "max_answer_tokens": dcfg.max_answer_tokens,
                        "max_new_tokens": dcfg.max_new_tokens,
                        "exp": exp,
                        "model_name_or_path": model_name_or_path,
                    }
                )
    return tasks


def _cmd_for_task(task: dict, log_json_path: Path) -> list[str]:
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--exp",
        task["exp"],
        "--model_name_or_path",
        task["model_name_or_path"],
        "--dataset_name",
        task["dataset"],
        "--split",
        "test",
        "--batch_size",
        "8",
        "--max_seq_len",
        "2048",
        "--max_neighbors_per_hop",
        str(task["max_neighbors_per_hop"]),
        "--max_hops",
        str(task["max_hops"]),
        "--steps",
        str(task["steps"]),
        "--temperature",
        "0.0",
        "--remasking",
        "low_confidence",
        "--prompt_layout",
        "target_first",
        "--prompt_format",
        "category_infill",
        "--use_chat_template",
        "False",
        "--max_answer_tokens",
        str(task["max_answer_tokens"]),
        "--max_new_tokens",
        str(task["max_new_tokens"]),
        "--use_topology_mask",
        "True",
        "--log_file",
        str(log_json_path),
    ]


def main() -> int:
    args = _parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / ".logs" / f"topo_nb_sweep_{run_id}"
    stdout_dir = run_dir / "stdout"
    records_dir = run_dir / "records"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(args.model_name_or_path)
    summary_path = run_dir / "launcher_summary.jsonl"
    queue_path = run_dir / "launcher_queue.jsonl"

    eligible_gpus = _detect_eligible_gpus(args.min_free_mem_mb, args.max_util_pct)
    if not eligible_gpus:
        raise RuntimeError("No GPUs detected via nvidia-smi.")

    max_parallel = max(1, min(args.max_parallel, len(eligible_gpus)))
    gpu_pool = eligible_gpus[:max_parallel]
    free_gpus = gpu_pool.copy()
    running: list[dict] = []

    header = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "max_parallel": max_parallel,
        "gpu_pool": gpu_pool,
        "num_tasks": len(tasks),
    }
    print(json.dumps(header), flush=True)
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "header", **header}) + "\n")

    pending = tasks.copy()
    num_failed = 0

    while pending or running:
        while pending and free_gpus:
            task = pending.pop(0)
            gpu = free_gpus.pop(0)
            stdout_path = stdout_dir / f"{task['exp']}.log"
            rec_path = records_dir / f"{task['exp']}.jsonl"
            cmd = _cmd_for_task(task, rec_path)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"

            fout = open(stdout_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=fout,
                stderr=subprocess.STDOUT,
                text=True,
            )
            launched = {
                "type": "launch",
                "timestamp": datetime.now().isoformat(),
                "pid": proc.pid,
                "gpu": gpu,
                "stdout_log": str(stdout_path),
                "record_log": str(rec_path),
                "task": task,
            }
            print(json.dumps(launched), flush=True)
            with queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(launched) + "\n")
            running.append(
                {
                    "proc": proc,
                    "gpu": gpu,
                    "stdout_handle": fout,
                    "task": task,
                    "stdout_log": str(stdout_path),
                    "record_log": str(rec_path),
                    "start_time": time.time(),
                }
            )

        # Poll running tasks
        time.sleep(5)
        still_running: list[dict] = []
        for item in running:
            proc: subprocess.Popen = item["proc"]
            ret = proc.poll()
            if ret is None:
                still_running.append(item)
                continue

            item["stdout_handle"].close()
            free_gpus.append(item["gpu"])
            free_gpus.sort()
            elapsed = time.time() - item["start_time"]
            done = {
                "type": "finish",
                "timestamp": datetime.now().isoformat(),
                "pid": proc.pid,
                "gpu": item["gpu"],
                "returncode": ret,
                "elapsed_seconds": round(elapsed, 1),
                "stdout_log": item["stdout_log"],
                "record_log": item["record_log"],
                "task": item["task"],
            }
            print(json.dumps(done), flush=True)
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(done) + "\n")
            if ret != 0:
                num_failed += 1
        running = still_running

    trailer = {
        "type": "trailer",
        "timestamp": datetime.now().isoformat(),
        "num_tasks": len(tasks),
        "num_failed": num_failed,
        "run_dir": str(run_dir),
    }
    print(json.dumps(trailer), flush=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trailer) + "\n")
    return 1 if num_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
