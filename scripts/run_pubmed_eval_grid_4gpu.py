"""
Run PubMed eval grid on 4 GPUs and save summary files.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_eval_grid_4gpu.py
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


@dataclass(frozen=True)
class Task:
    eval_type: str  # "infill" | "logit"
    max_hops: int   # 1 | 2 | 3
    use_topology_mask: bool

    @property
    def topo_tag(self) -> str:
        return "topo1" if self.use_topology_mask else "topo0"

    @property
    def exp_name(self) -> str:
        return (
            f"pubmed_{self.eval_type}_{self.max_hops}hop_"
            f"{self.topo_tag}_nb10_catinfill_targetfirst"
        )


def build_cmd(task: Task, model_path: str, log_file: Path) -> list[str]:
    common = [
        "--model_name_or_path",
        model_path,
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
        str(task.max_hops),
        "--prompt_format",
        "category_infill",
        "--prompt_layout",
        "target_first",
        "--seed",
        "42",
        "--use_topology_mask",
        "True" if task.use_topology_mask else "False",
        "--exp",
        task.exp_name,
        "--log_file",
        str(log_file),
    ]
    if task.eval_type == "infill":
        return [
            "python",
            "examples/tmdlm/eval_infill.py",
            *common,
            "--steps",
            "10",
            "--temperature",
            "0.0",
            "--remasking",
            "low_confidence",
        ]
    if task.eval_type == "logit":
        return [
            "python",
            "examples/tmdlm/eval_logit.py",
            *common,
        ]
    raise ValueError(f"Unknown eval_type: {task.eval_type}")


def read_last_jsonl(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> None:
    repo_root = Path("/home/lingjie7/auto-research/projects/dlm-graph")
    model_path = "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
    gpus = [0, 1, 2, 3]  # hard cap: 4 GPUs only

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = repo_root / ".logs" / f"pubmed_eval_grid_{now}"
    run_dir = repo_root / "experiments" / f"pubmed_eval_grid_{now}"
    summaries_dir = repo_root / "summaries"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        Task(eval_type=eval_type, max_hops=h, use_topology_mask=topo)
        for eval_type in ("infill", "logit")
        for h in (1, 2, 3)
        for topo in (False, True)
    ]

    queue = list(tasks)
    available = list(gpus)
    running: list[dict] = []
    failures: list[tuple[Task, int]] = []

    print(f"[start] {len(tasks)} tasks, GPUs={gpus}", flush=True)
    print(f"[logs]  {logs_dir}", flush=True)
    print(f"[json]  {run_dir}", flush=True)

    while queue or running:
        while queue and available:
            task = queue.pop(0)
            gpu = available.pop(0)
            jsonl_path = run_dir / f"{task.exp_name}.jsonl"
            out_path = logs_dir / f"{task.exp_name}.out"
            cmd = build_cmd(task, model_path, jsonl_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(out_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
            running.append(
                {
                    "proc": proc,
                    "fh": fh,
                    "task": task,
                    "gpu": gpu,
                    "out_path": out_path,
                    "jsonl_path": jsonl_path,
                    "start_ts": time.time(),
                }
            )
            print(f"[launch] gpu={gpu} {task.exp_name}", flush=True)

        time.sleep(10)
        still_running: list[dict] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue

            item["fh"].close()
            elapsed = time.time() - item["start_ts"]
            task = item["task"]
            gpu = item["gpu"]
            if ret == 0:
                print(f"[done]   gpu={gpu} {task.exp_name} ({elapsed:.1f}s)", flush=True)
            else:
                print(f"[fail]   gpu={gpu} {task.exp_name} ret={ret}", flush=True)
                failures.append((task, ret))
            available.append(gpu)
        running = still_running

    # Build summary artifacts.
    rows = []
    for task in tasks:
        rec = read_last_jsonl(run_dir / f"{task.exp_name}.jsonl")
        row = {
            "experiment": task.exp_name,
            "eval_type": task.eval_type,
            "max_hops": task.max_hops,
            "use_topology_mask": task.use_topology_mask,
            "accuracy": rec.get("accuracy"),
            "accuracy_strict": rec.get("accuracy_strict"),
            "accuracy_lenient": rec.get("accuracy_lenient"),
            "elapsed_seconds": rec.get("elapsed_seconds"),
            "jsonl_path": str(run_dir / f"{task.exp_name}.jsonl"),
            "status": "ok" if rec else "missing",
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["eval_type"], r["max_hops"], r["use_topology_mask"]))

    json_summary = summaries_dir / f"pubmed_eval_grid_summary_{now}.json"
    json_summary.write_text(json.dumps(rows, indent=2))

    csv_summary = summaries_dir / f"pubmed_eval_grid_summary_{now}.csv"
    with open(csv_summary, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "eval_type",
                "max_hops",
                "use_topology_mask",
                "accuracy",
                "accuracy_strict",
                "accuracy_lenient",
                "elapsed_seconds",
                "jsonl_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_summary = summaries_dir / f"pubmed_eval_grid_summary_{now}.md"
    md_lines = [
        "# PubMed Eval Grid Summary",
        "",
        f"- Timestamp: `{now}`",
        f"- Model: `{model_path}`",
        "- Fixed settings: `prompt_format=category_infill`, `prompt_layout=target_first`, `max_neighbors_per_hop=10`, `max_seq_len=2048`, `batch_size=1`",
        "- Variables: `eval_type in {infill, logit}`, `max_hops in {1,2,3}`, `use_topology_mask in {False,True}`",
        "- GPU budget: exactly 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`)",
        "",
        "## Results",
        "",
        "| eval_type | hops | topo_mask | accuracy | accuracy_strict | accuracy_lenient | elapsed_s | status |",
        "|---|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            "| "
            f"{r['eval_type']} | "
            f"{r['max_hops']} | "
            f"{r['use_topology_mask']} | "
            f"{'' if r['accuracy'] is None else r['accuracy']} | "
            f"{'' if r['accuracy_strict'] is None else r['accuracy_strict']} | "
            f"{'' if r['accuracy_lenient'] is None else r['accuracy_lenient']} | "
            f"{'' if r['elapsed_seconds'] is None else r['elapsed_seconds']} | "
            f"{r['status']} |"
        )
    md_lines += [
        "",
        "## Artifacts",
        "",
        f"- Per-run jsonl: `{run_dir}`",
        f"- Per-run stdout: `{logs_dir}`",
        f"- JSON summary: `{json_summary}`",
        f"- CSV summary: `{csv_summary}`",
    ]
    md_summary.write_text("\n".join(md_lines) + "\n")

    print(f"[summary] {md_summary}", flush=True)
    print(f"[summary] {csv_summary}", flush=True)
    print(f"[summary] {json_summary}", flush=True)

    if failures:
        print(f"[final] completed with {len(failures)} failed tasks", flush=True)
        raise SystemExit(1)

    print("[final] all tasks completed successfully", flush=True)


if __name__ == "__main__":
    main()
