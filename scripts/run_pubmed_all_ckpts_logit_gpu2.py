"""
Run PubMed checkpoint eval_logit only on a single GPU.

This script re-evaluates all PubMed checkpoints with the current eval_logit.py
implementation, intended for the hierarchical PubMed decision rule.

Run:
    /home/lingjie7/anaconda3/envs/dllm/bin/python /home/lingjie7/auto-research/projects/dlm-graph/scripts/run_pubmed_all_ckpts_logit_gpu2.py
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
PYTHON_EXECUTABLE = os.environ.get(
    "PYTHON_EXECUTABLE",
    "/home/lingjie7/anaconda3/envs/dllm/bin/python",
)
MODEL_NAME_OR_PATH = os.environ.get(
    "MODEL_NAME_OR_PATH",
    "/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct",
)
TARGET_GPU = int(os.environ.get("TARGET_GPU", "2"))


@dataclass(frozen=True)
class CkptSpec:
    checkpoint_path: Path
    run_dir_name: str
    checkpoint_name: str
    max_hops: int
    use_topology_mask: bool
    topo_tag: str
    step_value: int

    @property
    def exp_name(self) -> str:
        return f"pubmed_logit_{self.run_dir_name}_{self.checkpoint_name}"


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


def build_cmd(spec: CkptSpec, jsonl_path: Path) -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        "examples/tmdlm/eval_logit.py",
        "--exp",
        spec.exp_name,
        "--model_name_or_path",
        MODEL_NAME_OR_PATH,
        "--lora_path",
        str(spec.checkpoint_path),
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
        str(spec.max_hops),
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
        "True" if spec.use_topology_mask else "False",
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


def main() -> None:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = SUMMARY_ROOT / f"pubmed_logit_hier_gpu{TARGET_GPU}_{now}"
    jsonl_dir = run_root / "jsonl"
    stdout_dir = run_root / "stdout"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)

    specs: list[CkptSpec] = []
    for path in sorted(MODELS_ROOT.glob("tmdlm-llada-8b-pubmed-*/checkpoint-*")):
        if not path.is_dir():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        spec = parse_ckpt_spec(path)
        if spec is not None:
            specs.append(spec)
    specs.sort(key=lambda s: (s.topo_tag, s.step_value, s.checkpoint_name))

    print(f"[start] checkpoints={len(specs)} gpu={TARGET_GPU}", flush=True)
    print(f"[jsonl] {jsonl_dir}", flush=True)
    print(f"[stdout] {stdout_dir}", flush=True)

    rows = []
    for spec in specs:
        jsonl_path = jsonl_dir / f"{spec.exp_name}.jsonl"
        stdout_path = stdout_dir / f"{spec.exp_name}.out"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(TARGET_GPU)
        cmd = build_cmd(spec, jsonl_path)
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
        rows.append(
            {
                "experiment": spec.exp_name,
                "run_name": spec.run_dir_name,
                "checkpoint_name": spec.checkpoint_name,
                "topo_tag": spec.topo_tag,
                "max_hops": spec.max_hops,
                "use_topology_mask": spec.use_topology_mask,
                "gpu": TARGET_GPU,
                "return_code": ret,
                "elapsed_wall_s": elapsed,
                "accuracy": rec.get("accuracy"),
                "elapsed_seconds": rec.get("elapsed_seconds"),
                "jsonl_path": str(jsonl_path),
                "stdout_path": str(stdout_path),
            }
        )
        status = "done" if ret == 0 else "fail"
        print(f"[{status}] {spec.exp_name} ret={ret} ({elapsed}s)", flush=True)

    csv_path = run_root / "summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "run_name",
                "checkpoint_name",
                "topo_tag",
                "max_hops",
                "use_topology_mask",
                "gpu",
                "return_code",
                "elapsed_wall_s",
                "accuracy",
                "elapsed_seconds",
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
