"""
Run Cora final-checkpoint eval_infill step sweep without GPU polling.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    TARGET_GPUS=0 TOPO_TAG=notopo python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_infill_steps.py
    TARGET_GPUS=1 TOPO_TAG=topo python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_infill_steps.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
MODELS_ROOT = REPO_ROOT / ".models"
SUMMARIES_ROOT = REPO_ROOT / "summaries"
MODEL_NAME = os.environ.get(
    "MODEL_NAME_OR_PATH", "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
)
TARGET_GPUS = [int(x) for x in os.environ.get("TARGET_GPUS", "2,6").split(",") if x]
STEPS_LIST = [1, 2, 4, 8]
TOPO_TAG = os.environ.get("TOPO_TAG", "both").strip().lower()


@dataclass(frozen=True)
class Task:
    topo_tag: str
    use_topology_mask: bool
    checkpoint_path: Path
    step: int

    @property
    def exp_name(self) -> str:
        return (
            "cora_infill_final_"
            f"{self.topo_tag}_steps{self.step}_nb10_catinfill_labelon"
        )


def discover_final_checkpoint(topo_tag: str) -> Path:
    pattern = (
        f"tmdlm-llada-8b-cora-2hop-{topo_tag}-"
        "catinfill-nbmask-noeospad-r64-ep20-*"
    )
    dirs = sorted(p for p in MODELS_ROOT.glob(pattern) if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No run dir found for pattern: {pattern}")
    ckpt = dirs[-1] / "checkpoint-final"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing checkpoint-final in {dirs[-1]}")
    if not (ckpt / "adapter_config.json").exists():
        raise FileNotFoundError(f"Invalid checkpoint: {ckpt}")
    return ckpt


def build_cmd(task: Task, jsonl_path: Path) -> list[str]:
    return [
        sys.executable,
        "examples/tmdlm/eval_infill.py",
        "--exp",
        task.exp_name,
        "--model_name_or_path",
        MODEL_NAME,
        "--lora_path",
        str(task.checkpoint_path),
        "--dataset_name",
        "cora",
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
        "6",
        "--max_new_tokens",
        "6",
        "--include_neighbor_labels",
        "True",
        "--neighbor_label_format",
        "bracket",
        "--use_topology_mask",
        "True" if task.use_topology_mask else "False",
        "--steps",
        str(task.step),
        "--temperature",
        "0.0",
        "--remasking",
        "low_confidence",
        "--seed",
        "42",
        "--log_file",
        str(jsonl_path),
    ]


def read_jsonl_tail(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> None:
    if not TARGET_GPUS:
        raise ValueError("TARGET_GPUS is empty.")
    if TOPO_TAG not in {"both", "topo", "notopo"}:
        raise ValueError("TOPO_TAG must be one of: both, topo, notopo.")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = SUMMARIES_ROOT / f"cora_final_infill_steps_direct_{now}"
    jsonl_dir = out_root / "jsonl"
    stdout_dir = out_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    ckpt_notopo = discover_final_checkpoint("notopo")
    ckpt_topo = discover_final_checkpoint("topo")

    tasks: list[Task] = []
    for step in STEPS_LIST:
        if TOPO_TAG in {"both", "notopo"}:
            tasks.append(Task("notopo", False, ckpt_notopo, step))
        if TOPO_TAG in {"both", "topo"}:
            tasks.append(Task("topo", True, ckpt_topo, step))

    queue = list(tasks)
    running: list[dict] = []
    finished: list[dict] = []

    print(
        f"[start] tasks_total={len(tasks)} target_gpus={TARGET_GPUS} topo_tag={TOPO_TAG}",
        flush=True,
    )
    print(f"[checkpoint:notopo] {ckpt_notopo}", flush=True)
    print(f"[checkpoint:topo]   {ckpt_topo}", flush=True)
    print(f"[jsonl]  {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    while queue or running:
        while queue and len(running) < len(TARGET_GPUS):
            gpu = TARGET_GPUS[len(running)]
            task = queue.pop(0)
            out_path = stdout_dir / f"{task.exp_name}.out"
            jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(out_path, "w")
            proc = subprocess.Popen(
                build_cmd(task, jsonl_path),
                cwd=REPO_ROOT,
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
                    "start_ts": time.time(),
                    "jsonl_path": jsonl_path,
                    "stdout_path": out_path,
                }
            )
            print(
                f"[launch] gpu={gpu} {task.exp_name} steps={task.step} topo={task.topo_tag}",
                flush=True,
            )

        time.sleep(5)

        still_running: list[dict] = []
        freed_gpus: list[int] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue

            item["fh"].close()
            freed_gpus.append(item["gpu"])
            elapsed = round(time.time() - item["start_ts"], 2)
            task: Task = item["task"]
            rec = read_jsonl_tail(item["jsonl_path"])
            finished.append(
                {
                    "experiment": task.exp_name,
                    "dataset": "cora",
                    "eval_type": "infill",
                    "topo_tag": task.topo_tag,
                    "use_topology_mask": task.use_topology_mask,
                    "checkpoint_path": str(task.checkpoint_path),
                    "checkpoint_name": "checkpoint-final",
                    "steps": task.step,
                    "max_hops": 2,
                    "max_neighbors_per_hop": 10,
                    "max_answer_tokens": 6,
                    "max_new_tokens": 6,
                    "include_neighbor_labels": True,
                    "neighbor_label_format": "bracket",
                    "gpu": item["gpu"],
                    "return_code": ret,
                    "elapsed_wall_s": elapsed,
                    "accuracy": rec.get("accuracy"),
                    "accuracy_strict": rec.get("accuracy_strict"),
                    "accuracy_lenient": rec.get("accuracy_lenient"),
                    "elapsed_seconds": rec.get("elapsed_seconds"),
                    "jsonl_path": str(item["jsonl_path"]),
                    "stdout_path": str(item["stdout_path"]),
                }
            )
            status = "done" if ret == 0 else "fail"
            print(
                f"[{status}] gpu={item['gpu']} {task.exp_name} ret={ret} ({elapsed}s)",
                flush=True,
            )

        if freed_gpus:
            running = still_running
            running.sort(key=lambda x: TARGET_GPUS.index(x["gpu"]))
        else:
            running = still_running

    finished.sort(key=lambda row: (row["topo_tag"], row["steps"]))
    json_path = out_root / "summary.json"
    json_path.write_text(json.dumps(finished, indent=2))

    csv_path = out_root / "summary.csv"
    fields = [
        "experiment",
        "dataset",
        "eval_type",
        "topo_tag",
        "use_topology_mask",
        "checkpoint_path",
        "checkpoint_name",
        "steps",
        "max_hops",
        "max_neighbors_per_hop",
        "max_answer_tokens",
        "max_new_tokens",
        "include_neighbor_labels",
        "neighbor_label_format",
        "gpu",
        "return_code",
        "elapsed_wall_s",
        "accuracy",
        "accuracy_strict",
        "accuracy_lenient",
        "elapsed_seconds",
        "jsonl_path",
        "stdout_path",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(finished)

    md_path = out_root / "summary.md"
    md_path.write_text(
        "\n".join(
            [
                "# Cora Final Checkpoint Infill Step Sweep",
                "",
                f"- Timestamp: `{now}`",
                f"- Target GPUs: `{TARGET_GPUS}`",
                f"- Checkpoint (notopo): `{ckpt_notopo}`",
                f"- Checkpoint (topo): `{ckpt_topo}`",
                "- Mode: direct launch without GPU polling",
                "",
                f"- Summary CSV: `{csv_path}`",
                f"- Summary JSON: `{json_path}`",
                f"- Per-run JSONL: `{jsonl_dir}`",
                f"- Per-run stdout: `{stdout_dir}`",
            ]
        )
        + "\n"
    )
    print(f"[summary] {md_path}", flush=True)
    print(f"[summary] {csv_path}", flush=True)
    print(f"[summary] {json_path}", flush=True)


if __name__ == "__main__":
    main()
