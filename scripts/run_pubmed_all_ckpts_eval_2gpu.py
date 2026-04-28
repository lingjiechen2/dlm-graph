"""
Run all PubMed checkpoints with both eval_logit and eval_infill on 2 GPUs.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_all_ckpts_eval_2gpu.py
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


@dataclass(frozen=True)
class CkptSpec:
    checkpoint_path: Path
    run_dir_name: str
    checkpoint_name: str
    max_hops: int
    use_topology_mask: bool
    step_value: int


@dataclass(frozen=True)
class Task:
    eval_type: str  # "logit" | "infill"
    ckpt: CkptSpec

    @property
    def exp_name(self) -> str:
        return (
            f"pubmed_{self.eval_type}_"
            f"{self.ckpt.run_dir_name}_{self.ckpt.checkpoint_name}"
        )


def parse_ckpt_spec(path: Path) -> CkptSpec | None:
    run_dir_name = path.parent.name
    checkpoint_name = path.name
    m = re.match(
        r"^tmdlm-llada-8b-pubmed-(?P<hops>[0-9]+hop|na)-(?P<topo>topo|notopo|na)-.*$",
        run_dir_name,
    )
    if m is None:
        return None

    hops_token = m.group("hops")
    topo_token = m.group("topo")
    max_hops = 0 if hops_token == "na" else int(hops_token.replace("hop", ""))
    use_topology_mask = topo_token == "topo"

    if checkpoint_name == "checkpoint-final":
        step_value = 10**9
    else:
        step_value = int(checkpoint_name.replace("checkpoint-", ""))

    return CkptSpec(
        checkpoint_path=path,
        run_dir_name=run_dir_name,
        checkpoint_name=checkpoint_name,
        max_hops=max_hops,
        use_topology_mask=use_topology_mask,
        step_value=step_value,
    )


def build_cmd(task: Task, model_name_or_path: str, jsonl_path: Path) -> list[str]:
    common = [
        "--exp",
        task.exp_name,
        "--model_name_or_path",
        model_name_or_path,
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
            "python",
            "examples/tmdlm/eval_logit.py",
            *common,
            "--position_id_type",
            "sequential",
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
            "--max_new_tokens",
            "6",
        ]
    raise ValueError(f"Unknown eval_type: {task.eval_type}")


def read_jsonl_tail(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> None:
    repo_root = Path("/home/lingjie7/auto-research/projects/dlm-graph")
    models_root = repo_root / ".models"
    summary_root = repo_root / "summaries"
    model_name_or_path = os.environ.get(
        "MODEL_NAME_OR_PATH",
        "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct",
    )
    gpu_list = [int(x) for x in os.environ.get("EVAL_GPUS", "0,1").split(",") if x]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = summary_root / f"pubmed_noeospad_allckpts_eval_gpu01_{now}"
    jsonl_dir = run_root / "jsonl"
    stdout_dir = run_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    ckpt_paths = sorted(models_root.glob("tmdlm-llada-8b-pubmed-*/checkpoint-*"))
    specs: list[CkptSpec] = []
    for p in ckpt_paths:
        if not p.is_dir():
            continue
        if not (p / "adapter_config.json").exists():
            continue
        spec = parse_ckpt_spec(p)
        if spec is None:
            continue
        specs.append(spec)

    specs.sort(key=lambda s: (s.run_dir_name, s.step_value, s.checkpoint_name))
    tasks = [Task(eval_type=et, ckpt=s) for s in specs for et in ("logit", "infill")]

    print(f"[start] specs={len(specs)} tasks={len(tasks)} gpus={gpu_list}", flush=True)
    print(f"[jsonl] {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    queue = list(tasks)
    available = list(gpu_list)
    running: list[dict] = []
    finished: list[dict] = []

    while queue or running:
        while queue and available:
            task = queue.pop(0)
            gpu = available.pop(0)
            out_path = stdout_dir / f"{task.exp_name}.out"
            jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
            cmd = build_cmd(task, model_name_or_path=model_name_or_path, jsonl_path=jsonl_path)
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

        time.sleep(8)
        still_running: list[dict] = []
        for item in running:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue

            item["fh"].close()
            elapsed = time.time() - item["start_ts"]
            task: Task = item["task"]
            record = {
                "experiment": task.exp_name,
                "eval_type": task.eval_type,
                "dataset": "pubmed",
                "checkpoint_path": str(task.ckpt.checkpoint_path),
                "run_name": task.ckpt.run_dir_name,
                "checkpoint_name": task.ckpt.checkpoint_name,
                "max_hops": task.ckpt.max_hops,
                "use_topology_mask": task.ckpt.use_topology_mask,
                "include_neighbor_labels": True,
                "neighbor_label_format": "bracket",
                "max_answer_tokens": 6,
                "max_new_tokens": 6 if task.eval_type == "infill" else None,
                "gpu": item["gpu"],
                "elapsed_wall_s": round(elapsed, 2),
                "return_code": ret,
                "jsonl_path": str(item["jsonl_path"]),
                "stdout_path": str(item["out_path"]),
            }
            rec = read_jsonl_tail(item["jsonl_path"])
            if rec:
                record["accuracy"] = rec.get("accuracy")
                record["accuracy_strict"] = rec.get("accuracy_strict")
                record["accuracy_lenient"] = rec.get("accuracy_lenient")
                record["elapsed_seconds"] = rec.get("elapsed_seconds")
                record["per_class_accuracy"] = rec.get("per_class_accuracy")
                record["per_class_accuracy_strict"] = rec.get("per_class_accuracy_strict")
                record["per_class_accuracy_lenient"] = rec.get("per_class_accuracy_lenient")
            finished.append(record)
            if ret == 0:
                print(f"[done] gpu={item['gpu']} {task.exp_name} ({elapsed:.1f}s)", flush=True)
            else:
                print(f"[fail] gpu={item['gpu']} {task.exp_name} ret={ret}", flush=True)
            available.append(item["gpu"])
        running = still_running

    finished.sort(key=lambda r: (r["run_name"], r["checkpoint_name"], r["eval_type"]))
    json_path = run_root / "summary.json"
    json_path.write_text(json.dumps(finished, indent=2))

    csv_path = run_root / "summary.csv"
    fieldnames = [
        "experiment",
        "eval_type",
        "dataset",
        "checkpoint_path",
        "run_name",
        "checkpoint_name",
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
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(finished)

    md_lines = [
        "# PubMed All Checkpoints Eval Summary",
        "",
        f"- Timestamp: `{now}`",
        f"- Specs discovered: `{len(specs)}`",
        f"- Tasks launched: `{len(tasks)}`",
        f"- GPUs: `{gpu_list}`",
        f"- Model: `{model_name_or_path}`",
        "- Fixed settings: `prompt_format=category_infill`, `prompt_layout=target_first`, `max_answer_tokens=6`, `max_new_tokens=6`, `include_neighbor_labels=True`",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{json_path}`",
        f"- Summary CSV: `{csv_path}`",
        f"- Per-run JSONL: `{jsonl_dir}`",
        f"- Per-run stdout: `{stdout_dir}`",
    ]
    (run_root / "summary.md").write_text("\n".join(md_lines) + "\n")
    print(f"[summary] {run_root / 'summary.md'}", flush=True)
    print(f"[summary] {json_path}", flush=True)
    print(f"[summary] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
