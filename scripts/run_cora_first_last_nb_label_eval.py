"""
Evaluate Cora checkpoints (first + last per run category) with neighbor labels enabled.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_cora_first_last_nb_label_eval.py
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
    run_dir_name: str
    checkpoint_path: Path
    checkpoint_name: str
    max_hops: int
    use_topology_mask: bool


@dataclass(frozen=True)
class Task:
    eval_type: str  # "logit" | "infill"
    ckpt: CkptSpec

    @property
    def exp_name(self) -> str:
        return (
            f"cora_nb_labels_{self.eval_type}_"
            f"{self.ckpt.run_dir_name}_{self.ckpt.checkpoint_name}"
        )


def parse_run_setting(run_dir_name: str) -> tuple[int, bool] | None:
    m = re.match(
        r"^tmdlm-llada-8b-cora-(?P<hops>[0-9]+hop|na)-(?P<topo>topo|notopo|na)-.*$",
        run_dir_name,
    )
    if m is None:
        return None
    hops_token = m.group("hops")
    topo_token = m.group("topo")
    max_hops = 0 if hops_token == "na" else int(hops_token.replace("hop", ""))
    use_topology_mask = topo_token == "topo"
    return max_hops, use_topology_mask


def get_first_last_checkpoints(run_dir: Path) -> list[Path]:
    ckpts = [
        p
        for p in run_dir.glob("checkpoint-*")
        if p.is_dir() and (p / "adapter_config.json").exists()
    ]
    if not ckpts:
        return []

    numeric: list[tuple[int, Path]] = []
    final: Path | None = None
    for p in ckpts:
        if p.name == "checkpoint-final":
            final = p
            continue
        suffix = p.name.replace("checkpoint-", "")
        if suffix.isdigit():
            numeric.append((int(suffix), p))

    numeric.sort(key=lambda x: x[0])
    first = numeric[0][1] if numeric else final
    last = final if final is not None else (numeric[-1][1] if numeric else first)

    chosen: list[Path] = []
    if first is not None:
        chosen.append(first)
    if last is not None and last != first:
        chosen.append(last)
    return chosen


def build_cmd(task: Task, model_name_or_path: str, log_file: Path) -> list[str]:
    common = [
        "--exp",
        task.exp_name,
        "--model_name_or_path",
        model_name_or_path,
        "--lora_path",
        str(task.ckpt.checkpoint_path),
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
        str(task.ckpt.max_hops),
        "--use_topology_mask",
        "True" if task.ckpt.use_topology_mask else "False",
        "--prompt_format",
        "mc_digit",
        "--max_answer_tokens",
        "1",
        "--include_neighbor_labels",
        "True",
        "--neighbor_label_format",
        "bracket",
        "--seed",
        "42",
        "--log_file",
        str(log_file),
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
            "1",
        ]
    raise ValueError(f"Unknown eval_type: {task.eval_type}")


def read_jsonl_last(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def main() -> None:
    repo_root = Path("/home/lingjie7/auto-research/projects/dlm-graph")
    models_root = repo_root / ".models"
    summaries_root = repo_root / "summaries"
    model_name_or_path = os.environ.get(
        "MODEL_NAME_OR_PATH", "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct"
    )
    gpus = [int(x) for x in os.environ.get("EVAL_GPUS", "0,1").split(",") if x]

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = summaries_root / f"cora_first_last_nb_labels_eval_{now}"
    jsonl_dir = out_dir / "jsonl"
    stdout_dir = out_dir / "stdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    specs: list[CkptSpec] = []
    for run_dir in sorted(models_root.glob("tmdlm-llada-8b-cora-*")):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_setting(run_dir.name)
        if parsed is None:
            continue
        max_hops, use_topology_mask = parsed
        selected = get_first_last_checkpoints(run_dir)
        for ckpt in selected:
            specs.append(
                CkptSpec(
                    run_dir_name=run_dir.name,
                    checkpoint_path=ckpt,
                    checkpoint_name=ckpt.name,
                    max_hops=max_hops,
                    use_topology_mask=use_topology_mask,
                )
            )

    specs.sort(key=lambda s: (s.run_dir_name, s.checkpoint_name))
    tasks = [Task(eval_type=et, ckpt=s) for s in specs for et in ("logit", "infill")]

    print(f"[start] categories={len(set(s.run_dir_name for s in specs))} ckpts={len(specs)} tasks={len(tasks)} gpus={gpus}", flush=True)
    print(f"[jsonl] {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    queue = list(tasks)
    available = list(gpus)
    running: list[dict] = []
    done: list[dict] = []

    while queue or running:
        while queue and available:
            task = queue.pop(0)
            gpu = available.pop(0)
            jsonl_path = jsonl_dir / f"{task.exp_name}.jsonl"
            stdout_path = stdout_dir / f"{task.exp_name}.out"
            cmd = build_cmd(task, model_name_or_path=model_name_or_path, log_file=jsonl_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            fh = open(stdout_path, "w")
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
                    "jsonl_path": jsonl_path,
                    "stdout_path": stdout_path,
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
            elapsed = round(time.time() - item["start_ts"], 2)
            task: Task = item["task"]
            record = {
                "experiment": task.exp_name,
                "eval_type": task.eval_type,
                "run_dir_name": task.ckpt.run_dir_name,
                "checkpoint_name": task.ckpt.checkpoint_name,
                "checkpoint_path": str(task.ckpt.checkpoint_path),
                "max_hops": task.ckpt.max_hops,
                "use_topology_mask": task.ckpt.use_topology_mask,
                "include_neighbor_labels": True,
                "neighbor_label_format": "bracket",
                "gpu": item["gpu"],
                "return_code": ret,
                "elapsed_wall_s": elapsed,
                "jsonl_path": str(item["jsonl_path"]),
                "stdout_path": str(item["stdout_path"]),
            }
            rec = read_jsonl_last(item["jsonl_path"])
            if rec:
                record["accuracy"] = rec.get("accuracy")
                record["accuracy_strict"] = rec.get("accuracy_strict")
                record["accuracy_lenient"] = rec.get("accuracy_lenient")
                record["elapsed_seconds"] = rec.get("elapsed_seconds")
            done.append(record)
            if ret == 0:
                print(f"[done] gpu={item['gpu']} {task.exp_name} ({elapsed}s)", flush=True)
            else:
                print(f"[fail] gpu={item['gpu']} {task.exp_name} ret={ret}", flush=True)
            available.append(item["gpu"])
        running = still_running

    done.sort(key=lambda r: (r["run_dir_name"], r["checkpoint_name"], r["eval_type"]))

    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(done, indent=2))

    fields = [
        "experiment",
        "eval_type",
        "run_dir_name",
        "checkpoint_name",
        "checkpoint_path",
        "max_hops",
        "use_topology_mask",
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
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(done)

    nonzero = sum(1 for r in done if r["return_code"] != 0)
    md_lines = [
        "# Cora First/Last Checkpoint Eval with Neighbor Labels",
        "",
        f"- Timestamp: `{now}`",
        f"- Categories: `{len(set(s.run_dir_name for s in specs))}`",
        f"- Checkpoints selected: `{len(specs)}`",
        f"- Tasks: `{len(tasks)}`",
        f"- GPUs: `{gpus}`",
        f"- Failures: `{nonzero}`",
        "",
        "## Artifacts",
        "",
        f"- Summary JSON: `{summary_json}`",
        f"- Summary CSV: `{summary_csv}`",
        f"- Per-run JSONL: `{jsonl_dir}`",
        f"- Per-run stdout: `{stdout_dir}`",
    ]
    summary_md.write_text("\n".join(md_lines) + "\n")
    print(f"[summary] {summary_md}", flush=True)
    print(f"[summary] {summary_csv}", flush=True)


if __name__ == "__main__":
    main()

