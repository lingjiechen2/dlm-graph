"""
Run all PubMed checkpoints with eval_logit and eval_infill on 4 GPUs.

This launcher evenly splits checkpoint specs across GPUs. Each GPU owns a fixed
subset of checkpoints and runs both eval_logit and eval_infill for each one.

Run:
    source ~/.zshrc
    conda activate ~/miniconda3/envs/dllm
    EVAL_GPUS=0,1,2,6 python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_all_ckpts_eval_4gpu_even.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/home/lingjie7/auto-research/projects/dlm-graph")
MODELS_ROOT = REPO_ROOT / ".models"
SUMMARY_ROOT = REPO_ROOT / "summaries"
MODEL_NAME_OR_PATH = os.environ.get(
    "MODEL_NAME_OR_PATH",
    "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct",
)
GPU_LIST = [int(x) for x in os.environ.get("EVAL_GPUS", "0,1,2,6").split(",") if x]
PYTHON_EXECUTABLE = os.environ.get(
    "PYTHON_EXECUTABLE",
    "/home/lingjie7/anaconda3/envs/dllm/bin/python",
)


@dataclass(frozen=True)
class CkptSpec:
    checkpoint_path: Path
    run_dir_name: str
    checkpoint_name: str
    max_hops: int
    use_topology_mask: bool
    topo_tag: str
    step_value: int


@dataclass(frozen=True)
class Task:
    eval_type: str
    ckpt: CkptSpec

    @property
    def exp_name(self) -> str:
        return f"pubmed_{self.eval_type}_{self.ckpt.run_dir_name}_{self.ckpt.checkpoint_name}"


def parse_ckpt_spec(path: Path) -> CkptSpec | None:
    run_dir_name = path.parent.name
    checkpoint_name = path.name
    match = re.match(
        r"^tmdlm-llada-8b-pubmed-(?P<hops>[0-9]+hop|na)-(?P<topo>topo|notopo|na)-.*$",
        run_dir_name,
    )
    if match is None:
        return None

    hops_token = match.group("hops")
    topo_tag = match.group("topo")
    max_hops = 0 if hops_token == "na" else int(hops_token.replace("hop", ""))
    use_topology_mask = topo_tag == "topo"
    step_value = 10**9 if checkpoint_name == "checkpoint-final" else int(
        checkpoint_name.replace("checkpoint-", "")
    )

    return CkptSpec(
        checkpoint_path=path,
        run_dir_name=run_dir_name,
        checkpoint_name=checkpoint_name,
        max_hops=max_hops,
        use_topology_mask=use_topology_mask,
        topo_tag=topo_tag,
        step_value=step_value,
    )


def build_cmd(task: Task, jsonl_path: Path) -> list[str]:
    common = [
        "--exp",
        task.exp_name,
        "--model_name_or_path",
        MODEL_NAME_OR_PATH,
        "--lora_path",
        str(task.ckpt.checkpoint_path),
        "--dataset_name",
        "pubmed",
        "--split",
        "test",
        "--batch_size",
        os.environ.get("EVAL_BATCH_SIZE", "1"),
        "--max_seq_len",
        "2048",
        "--max_neighbors_per_hop",
        "10",
        "--max_hops",
        str(task.ckpt.max_hops),
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
        "True" if task.ckpt.use_topology_mask else "False",
        "--seed",
        "42",
        "--log_file",
        str(jsonl_path),
    ]
    if task.eval_type == "logit":
        return [
            PYTHON_EXECUTABLE,
            "examples/tmdlm/eval_logit.py",
            *common,
            "--position_id_type",
            "sequential",
        ]
    if task.eval_type == "infill":
        return [
            PYTHON_EXECUTABLE,
            "examples/tmdlm/eval_infill.py",
            *common,
            "--steps",
            "10",
            "--temperature",
            "0.0",
            "--remasking",
            "low_confidence",
            "--max_new_tokens",
            "6",
        ]
    raise ValueError(f"Unknown eval_type: {task.eval_type}")


def read_jsonl_tail(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def split_evenly(specs: list[CkptSpec], gpus: list[int]) -> dict[int, list[CkptSpec]]:
    buckets = {gpu: [] for gpu in gpus}
    for idx, spec in enumerate(specs):
        gpu = gpus[idx % len(gpus)]
        buckets[gpu].append(spec)
    return buckets


def main() -> None:
    if not GPU_LIST:
        raise ValueError("EVAL_GPUS is empty.")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = SUMMARY_ROOT / f"pubmed_allckpts_eval_gpu0126_even_{now}"
    jsonl_dir = run_root / "jsonl"
    stdout_dir = run_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    ckpt_specs: list[CkptSpec] = []
    for path in sorted(MODELS_ROOT.glob("tmdlm-llada-8b-pubmed-*/checkpoint-*")):
        if not path.is_dir():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        spec = parse_ckpt_spec(path)
        if spec is not None:
            ckpt_specs.append(spec)

    ckpt_specs.sort(
        key=lambda spec: (spec.topo_tag, spec.step_value, spec.checkpoint_name, spec.run_dir_name)
    )
    gpu_to_specs = split_evenly(ckpt_specs, GPU_LIST)

    print(f"[start] specs={len(ckpt_specs)} gpus={GPU_LIST}", flush=True)
    print(f"[jsonl] {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)
    for gpu in GPU_LIST:
        names = [f"{s.run_dir_name}/{s.checkpoint_name}" for s in gpu_to_specs[gpu]]
        print(f"[assign] gpu={gpu} checkpoints={len(names)}", flush=True)
        for name in names:
            print(f"  - {name}", flush=True)

    workers: list[dict] = []
    for gpu in GPU_LIST:
        tasks: list[Task] = []
        for spec in gpu_to_specs[gpu]:
            for eval_type in ("logit", "infill"):
                tasks.append(Task(eval_type=eval_type, ckpt=spec))
        workers.append({"gpu": gpu, "queue": tasks, "running": None})

    finished: list[dict] = []

    while True:
        active = False
        for worker in workers:
            item = worker["running"]
            if item is not None:
                active = True
                ret = item["proc"].poll()
                if ret is None:
                    continue

                item["fh"].close()
                task: Task = item["task"]
                elapsed = round(time.time() - item["start_ts"], 2)
                rec = read_jsonl_tail(item["jsonl_path"])
                row = {
                    "experiment": task.exp_name,
                    "eval_type": task.eval_type,
                    "dataset": "pubmed",
                    "checkpoint_path": str(task.ckpt.checkpoint_path),
                    "run_name": task.ckpt.run_dir_name,
                    "checkpoint_name": task.ckpt.checkpoint_name,
                    "topo_tag": task.ckpt.topo_tag,
                    "max_hops": task.ckpt.max_hops,
                    "use_topology_mask": task.ckpt.use_topology_mask,
                    "include_neighbor_labels": True,
                    "neighbor_label_format": "bracket",
                    "max_answer_tokens": 6,
                    "max_new_tokens": 6 if task.eval_type == "infill" else None,
                    "gpu": worker["gpu"],
                    "elapsed_wall_s": elapsed,
                    "return_code": ret,
                    "jsonl_path": str(item["jsonl_path"]),
                    "stdout_path": str(item["stdout_path"]),
                    "accuracy": rec.get("accuracy"),
                    "accuracy_strict": rec.get("accuracy_strict"),
                    "accuracy_lenient": rec.get("accuracy_lenient"),
                    "elapsed_seconds": rec.get("elapsed_seconds"),
                }
                finished.append(row)
                status = "done" if ret == 0 else "fail"
                print(
                    f"[{status}] gpu={worker['gpu']} {task.exp_name} ret={ret} ({elapsed}s)",
                    flush=True,
                )
                worker["running"] = None

            if worker["running"] is None and worker["queue"]:
                active = True
                task = worker["queue"].pop(0)
                jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
                stdout_path = stdout_dir / f"{task.exp_name}.out"
                if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                    rec = read_jsonl_tail(jsonl_path)
                    finished.append(
                        {
                            "experiment": task.exp_name,
                            "eval_type": task.eval_type,
                            "dataset": "pubmed",
                            "checkpoint_path": str(task.ckpt.checkpoint_path),
                            "run_name": task.ckpt.run_dir_name,
                            "checkpoint_name": task.ckpt.checkpoint_name,
                            "topo_tag": task.ckpt.topo_tag,
                            "max_hops": task.ckpt.max_hops,
                            "use_topology_mask": task.ckpt.use_topology_mask,
                            "include_neighbor_labels": True,
                            "neighbor_label_format": "bracket",
                            "max_answer_tokens": 6,
                            "max_new_tokens": 6 if task.eval_type == "infill" else None,
                            "gpu": worker["gpu"],
                            "elapsed_wall_s": None,
                            "return_code": 0,
                            "jsonl_path": str(jsonl_path),
                            "stdout_path": str(stdout_path),
                            "accuracy": rec.get("accuracy"),
                            "accuracy_strict": rec.get("accuracy_strict"),
                            "accuracy_lenient": rec.get("accuracy_lenient"),
                            "elapsed_seconds": rec.get("elapsed_seconds"),
                        }
                    )
                    print(f"[skip] gpu={worker['gpu']} {task.exp_name}", flush=True)
                    continue

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(worker["gpu"])
                fh = open(stdout_path, "w")
                proc = subprocess.Popen(
                    build_cmd(task, jsonl_path=jsonl_path),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                )
                worker["running"] = {
                    "proc": proc,
                    "fh": fh,
                    "task": task,
                    "start_ts": time.time(),
                    "jsonl_path": jsonl_path,
                    "stdout_path": stdout_path,
                }
                print(f"[launch] gpu={worker['gpu']} {task.exp_name}", flush=True)

        if not active:
            break
        time.sleep(8)

    finished.sort(key=lambda row: (row["gpu"], row["run_name"], row["checkpoint_name"], row["eval_type"]))
    json_path = run_root / "summary.json"
    json_path.write_text(json.dumps(finished, indent=2))

    csv_path = run_root / "summary.csv"
    fields = [
        "experiment",
        "eval_type",
        "dataset",
        "checkpoint_path",
        "run_name",
        "checkpoint_name",
        "topo_tag",
        "max_hops",
        "use_topology_mask",
        "include_neighbor_labels",
        "neighbor_label_format",
        "max_answer_tokens",
        "max_new_tokens",
        "gpu",
        "elapsed_wall_s",
        "return_code",
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

    md_path = run_root / "summary.md"
    md_lines = [
        "# PubMed All Checkpoints Eval (4GPU Even Split)",
        "",
        f"- Timestamp: `{now}`",
        f"- GPUs: `{GPU_LIST}`",
        f"- Model: `{MODEL_NAME_OR_PATH}`",
        f"- Python: `{PYTHON_EXECUTABLE}`",
        "- Per-checkpoint task order: `eval_logit` then `eval_infill`",
        "- Fixed eval args: `dataset=pubmed`, `max_hops=2`, `max_neighbors_per_hop=10`, "
        "`prompt_format=category_infill`, `prompt_layout=target_first`, "
        "`max_answer_tokens=6`, `max_new_tokens=6`, "
        "`include_neighbor_labels=True`, `neighbor_label_format=bracket`, `infill_steps=10`",
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
