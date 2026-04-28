"""Run PubMed frozen-base eval_logit mask-length sweep on three GPUs.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_frozen_mask_sweep_3gpu.py

Optional env vars:
    TARGET_GPUS=0,1,6
    MODEL_NAME_OR_PATH=/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct
    PYTHON_EXECUTABLE=/home/lingjie7/anaconda3/envs/dllm/bin/python
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock, Thread


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
SUMMARY_ROOT = REPO_ROOT / "summaries"
PYTHON_EXECUTABLE = os.environ.get(
    "PYTHON_EXECUTABLE",
    "/home/lingjie7/anaconda3/envs/dllm/bin/python",
)
MODEL_NAME_OR_PATH = os.environ.get(
    "MODEL_NAME_OR_PATH",
    "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct",
)
TARGET_GPUS = [int(x) for x in os.environ.get("TARGET_GPUS", "0,1,6").split(",") if x]

OLD_EVAL = Path("/home/lingjie7/tmp/eval_logit_old_mask_sweep.py")
NEW_EVAL = Path("/home/lingjie7/tmp/eval_logit_new_mask_sweep.py")


@dataclass(frozen=True)
class Task:
    eval_tag: str
    eval_script: Path
    mask_tokens: int

    @property
    def exp_name(self) -> str:
        return f"pubmed_frozen_notopo_{self.eval_tag}_mask{self.mask_tokens}"


def build_cmd(task: Task, jsonl_path: Path) -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        str(task.eval_script),
        "--exp",
        task.exp_name,
        "--model_name_or_path",
        MODEL_NAME_OR_PATH,
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
        "--prompt_format",
        "category_infill",
        "--prompt_layout",
        "target_first",
        "--max_answer_tokens",
        str(task.mask_tokens),
        "--include_neighbor_labels",
        "True",
        "--neighbor_label_format",
        "bracket",
        "--use_topology_mask",
        "False",
        "--position_id_type",
        "sequential",
        "--seed",
        "42",
        "--log_file",
        str(jsonl_path),
    ]


def read_jsonl_tail(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def worker(gpu_id: int, queue: Queue, rows: list[dict], rows_lock: Lock, jsonl_dir: Path, stdout_dir: Path) -> None:
    while True:
        task = queue.get()
        if task is None:
            queue.task_done()
            return

        jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
        stdout_path = stdout_dir / f"{task.exp_name}.out"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = build_cmd(task, jsonl_path)
        start_ts = time.time()
        with open(stdout_path, "w") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            ret = proc.wait()
        elapsed = round(time.time() - start_ts, 2)
        rec = read_jsonl_tail(jsonl_path)
        row = {
            "experiment": task.exp_name,
            "eval_tag": task.eval_tag,
            "mask_tokens": task.mask_tokens,
            "gpu": gpu_id,
            "return_code": ret,
            "accuracy": rec.get("accuracy"),
            "elapsed_seconds": rec.get("elapsed_seconds"),
            "jsonl_path": str(jsonl_path),
            "stdout_path": str(stdout_path),
            "wall_s": elapsed,
        }
        with rows_lock:
            rows.append(row)
        status = "done" if ret == 0 else "fail"
        print(
            f"[{status}] gpu={gpu_id} {task.exp_name} ret={ret} ({elapsed}s) acc={row['accuracy']}",
            flush=True,
        )
        queue.task_done()


def main() -> None:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = SUMMARY_ROOT / f"pubmed_frozen_mask_sweep_3gpu_{now}"
    jsonl_dir = run_root / "jsonl"
    stdout_dir = run_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        Task(eval_tag=eval_tag, eval_script=eval_script, mask_tokens=mask_tokens)
        for eval_tag, eval_script in (("old", OLD_EVAL), ("new", NEW_EVAL))
        for mask_tokens in (3, 4, 5, 6)
    ]
    tasks.sort(key=lambda t: (t.mask_tokens, t.eval_tag))

    print(f"[start] tasks={len(tasks)} gpus={TARGET_GPUS}", flush=True)
    print(f"[jsonl] {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    queue: Queue = Queue()
    rows: list[dict] = []
    rows_lock = Lock()
    threads: list[Thread] = []

    for gpu_id in TARGET_GPUS:
        thread = Thread(
            target=worker,
            args=(gpu_id, queue, rows, rows_lock, jsonl_dir, stdout_dir),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for task in tasks:
        queue.put(task)
    for _ in TARGET_GPUS:
        queue.put(None)

    queue.join()
    for thread in threads:
        thread.join()

    rows.sort(key=lambda row: (row["mask_tokens"], row["eval_tag"]))
    csv_path = run_root / "summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "eval_tag",
                "mask_tokens",
                "gpu",
                "return_code",
                "accuracy",
                "elapsed_seconds",
                "wall_s",
                "jsonl_path",
                "stdout_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (run_root / "summary.json").write_text(json.dumps(rows, indent=2))
    print(f"[summary] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
