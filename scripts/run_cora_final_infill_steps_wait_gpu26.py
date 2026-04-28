"""
Run Cora final-checkpoint eval_infill step sweep with GPU polling.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_final_infill_steps_wait_gpu26.py
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


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
MODELS_ROOT = REPO_ROOT / ".models"
SUMMARIES_ROOT = REPO_ROOT / "summaries"
MODEL_NAME = os.environ.get(
    "MODEL_NAME_OR_PATH", "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
)
TARGET_GPUS = [int(x) for x in os.environ.get("TARGET_GPUS", "2,6").split(",") if x]
STEPS_LIST = [1, 2, 4, 8]


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
    run_dir = dirs[-1]
    ckpt = run_dir / "checkpoint-final"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing checkpoint-final in {run_dir}")
    if not (ckpt / "adapter_config.json").exists():
        raise FileNotFoundError(f"Invalid checkpoint: {ckpt}")
    return ckpt


def get_gpu_uuid_map() -> dict[str, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    mapping: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx_s, uuid = [x.strip() for x in line.split(",", maxsplit=1)]
        mapping[uuid] = int(idx_s)
    return mapping


def get_busy_gpus() -> set[int]:
    uuid_to_idx = get_gpu_uuid_map()
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    busy: set[int] = set()
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",", maxsplit=1)]
        if len(parts) < 2:
            continue
        uuid = parts[0]
        pid_s = parts[1]
        if not pid_s.isdigit():
            continue
        if not Path(f"/proc/{pid_s}").exists():
            continue
        idx = uuid_to_idx.get(uuid)
        if idx is not None:
            busy.add(idx)
    return busy


def build_cmd(task: Task, jsonl_path: Path) -> list[str]:
    return [
        "python",
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
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = SUMMARIES_ROOT / f"cora_final_infill_steps_gpu26_wait_{now}"
    jsonl_dir = out_root / "jsonl"
    stdout_dir = out_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    ckpt_notopo = discover_final_checkpoint("notopo")
    ckpt_topo = discover_final_checkpoint("topo")

    tasks: list[Task] = []
    for step in STEPS_LIST:
        tasks.append(
            Task(
                topo_tag="notopo",
                use_topology_mask=False,
                checkpoint_path=ckpt_notopo,
                step=step,
            )
        )
        tasks.append(
            Task(
                topo_tag="topo",
                use_topology_mask=True,
                checkpoint_path=ckpt_topo,
                step=step,
            )
        )

    finished: list[dict] = []
    queue: list[Task] = []
    for task in tasks:
        existing_jsonl = jsonl_dir / f"{task.exp_name}.jsonl"
        if existing_jsonl.exists() and existing_jsonl.stat().st_size > 0:
            rec = read_jsonl_tail(existing_jsonl)
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
                    "gpu": None,
                    "return_code": 0,
                    "elapsed_wall_s": None,
                    "accuracy": rec.get("accuracy"),
                    "accuracy_strict": rec.get("accuracy_strict"),
                    "accuracy_lenient": rec.get("accuracy_lenient"),
                    "elapsed_seconds": rec.get("elapsed_seconds"),
                    "jsonl_path": str(existing_jsonl),
                    "stdout_path": str(stdout_dir / f"{task.exp_name}.out"),
                }
            )
            continue
        queue.append(task)

    running: list[dict] = []
    print(
        f"[start] tasks_total={len(tasks)} pending={len(queue)} target_gpus={TARGET_GPUS}",
        flush=True,
    )
    print(f"[checkpoint:notopo] {ckpt_notopo}", flush=True)
    print(f"[checkpoint:topo]   {ckpt_topo}", flush=True)
    print(f"[jsonl]  {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    while queue or running:
        busy_gpus = get_busy_gpus()
        used_by_this = {item["gpu"] for item in running}
        free_gpus = [g for g in TARGET_GPUS if g not in busy_gpus and g not in used_by_this]

        while queue and free_gpus:
            task = queue.pop(0)
            gpu = free_gpus.pop(0)
            out_path = stdout_dir / f"{task.exp_name}.out"
            jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
            cmd = build_cmd(task, jsonl_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(out_path, "w")
            proc = subprocess.Popen(
                cmd,
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

        time.sleep(12)

        still_running: list[dict] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue

            item["fh"].close()
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
        running = still_running

        if queue and not running:
            busy_gpus = get_busy_gpus()
            print(
                f"[poll] waiting free gpu among {TARGET_GPUS}; "
                f"busy={sorted(list(busy_gpus))}; remaining_tasks={len(queue)}",
                flush=True,
            )

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
    md_lines = [
        "# Cora Final Checkpoint Infill Step Sweep",
        "",
        f"- Timestamp: `{now}`",
        f"- Target GPUs: `{TARGET_GPUS}` (polled until empty)",
        f"- Checkpoint (notopo): `{ckpt_notopo}`",
        f"- Checkpoint (topo): `{ckpt_topo}`",
        "- Fixed settings: `max_hops=2`, `max_neighbors_per_hop=10`, "
        "`prompt_format=category_infill`, `prompt_layout=target_first`, "
        "`max_answer_tokens=6`, `max_new_tokens=6`, "
        "`include_neighbor_labels=True`, `neighbor_label_format=bracket`",
        "- Variable: `steps in {1,2,4,8}`",
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{csv_path}`",
        f"- Summary JSON: `{json_path}`",
        f"- Per-run JSONL: `{jsonl_dir}`",
        f"- Per-run stdout: `{stdout_dir}`",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[summary] {md_path}", flush=True)
    print(f"[summary] {csv_path}", flush=True)
    print(f"[summary] {json_path}", flush=True)


if __name__ == "__main__":
    main()
