"""
Wait for Cora/PubMed frozen topo evals, update results.md, then launch ogbn-arxiv frozen evals.

Run from anywhere:
    /home/lingjie7/anaconda3/envs/dllm/bin/python \
        /home/lingjie7/auto-research/projects/dlm-graph/scripts/watch_topo_then_run_ogbn_arxiv.py

This script expects the Cora/PubMed topo JSONL files to be produced by the
currently running tmux eval jobs. Once both exist, it updates results.md and
launches ogbn-arxiv no-topo on GPU3 and topo on GPU7. Each ogbn-arxiv job
restarts sample_gen.py on its GPU after eval exits.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

BASE = Path("/home/lingjie7/auto-research/projects/dlm-graph")
PY = Path("/home/lingjie7/anaconda3/envs/dllm/bin/python")
MODEL = Path("/home/lingjie7/models/huggingface/GSAI-ML/LLaDA-8B-Instruct")
SAMPLE_GEN = Path("/home/lingjie7/sample_gen.py")
RESULTS_MD = BASE / "results.md"

CORA_JSON = BASE / "summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_topo_logit_labelon.jsonl"
CORA_OUT = BASE / "summaries/cora_llaga_frozen_eval_20260427/cora_llaga_frozen_topo_logit_labelon.out"
PUB_JSON = BASE / "summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_topo_logit_labelon.jsonl"
PUB_OUT = BASE / "summaries/pubmed_llaga_frozen_eval_20260427/pubmed_llaga_frozen_topo_logit_labelon.out"

OGBN_DIR = BASE / "summaries/ogbn_arxiv_llaga_frozen_eval_20260427"
OGBN_NOTOPO_JSON = OGBN_DIR / "ogbn_arxiv_llaga_frozen_notopo_logit_labelon.jsonl"
OGBN_NOTOPO_OUT = OGBN_DIR / "ogbn_arxiv_llaga_frozen_notopo_logit_labelon.out"
OGBN_TOPO_JSON = OGBN_DIR / "ogbn_arxiv_llaga_frozen_topo_logit_labelon.jsonl"
OGBN_TOPO_OUT = OGBN_DIR / "ogbn_arxiv_llaga_frozen_topo_logit_labelon.out"


def read_last_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def fmt_per_class(per_class: dict[str, Any]) -> str:
    return "; ".join(f"{k}: {float(v):.2f}" for k, v in per_class.items())


def replace_or_insert_table_row(text: str, row: str, dataset: str, topo: bool) -> str:
    lines = text.splitlines()
    marker = f"| 2026-04-27 | {dataset} | test | eval_logit | {str(topo)} | True |"
    for i, line in enumerate(lines):
        if line.startswith(marker):
            lines[i] = row
            return "\n".join(lines) + "\n"
    table_header = "| Date | Dataset | Split | Method | Topology Mask | Neighbor Labels | Accuracy | Per-class Accuracy | Output |"
    for i, line in enumerate(lines):
        if line.strip() == table_header:
            # Insert after separator and existing frozen rows.
            j = i + 2
            while j < len(lines) and lines[j].startswith("| 2026-04-27 |"):
                j += 1
            lines.insert(j, row)
            return "\n".join(lines) + "\n"
    return text.rstrip() + "\n" + row + "\n"


def append_details_if_missing(text: str, title: str, body: str) -> str:
    if title in text:
        return text
    marker = "## Running / Pending\n"
    if marker in text:
        return text.replace(marker, body.rstrip() + "\n\n" + marker)
    return text.rstrip() + "\n\n" + body.rstrip() + "\n"


def update_results(cora: dict[str, Any], pub: dict[str, Any]) -> None:
    text = RESULTS_MD.read_text() if RESULTS_MD.exists() else "# DLM-Graph Results\n\n"
    rows = []
    for data, path in [(cora, CORA_JSON), (pub, PUB_JSON)]:
        row = (
            f"| 2026-04-27 | {data['dataset'].capitalize() if data['dataset'] != 'pubmed' else 'PubMed'} "
            f"| {data['split']} | eval_logit | {data['config']['use_topology_mask']} | "
            f"{data['config']['include_neighbor_labels']} | {float(data['accuracy']):.2f} | "
            f"{fmt_per_class(data['per_class_accuracy'])} | `{path}` |"
        )
        rows.append((data["dataset"].capitalize() if data["dataset"] != "pubmed" else "PubMed", data['config']['use_topology_mask'], row))
    for dataset, topo, row in rows:
        text = replace_or_insert_table_row(text, row, dataset, bool(topo))

    cora_body = f"""### Cora Frozen Base Topo Details

- Experiment: `{cora['experiment']}`
- Model: `{cora['model']}`
- Dataset: `cora`
- Split: `test`
- Accuracy: `{float(cora['accuracy']):.2f}%`
- Elapsed time: `{float(cora['elapsed_seconds']):.1f}s`
- Per-class accuracy: `{fmt_per_class(cora['per_class_accuracy'])}`
- Stdout: `{CORA_OUT}`
- JSONL: `{CORA_JSON}`
"""
    pub_body = f"""### PubMed Frozen Base Topo Details

- Experiment: `{pub['experiment']}`
- Model: `{pub['model']}`
- Dataset: `pubmed`
- Split: `test`
- Accuracy: `{float(pub['accuracy']):.2f}%`
- Elapsed time: `{float(pub['elapsed_seconds']):.1f}s`
- Per-class accuracy: `{fmt_per_class(pub['per_class_accuracy'])}`
- Stdout: `{PUB_OUT}`
- JSONL: `{PUB_JSON}`
"""
    text = append_details_if_missing(text, "### Cora Frozen Base Topo Details", cora_body)
    text = append_details_if_missing(text, "### PubMed Frozen Base Topo Details", pub_body)

    pending_rows = [
        (
            "| 2026-04-27 | ogbn-arxiv | test | eval_logit | False | True | pending | pending | "
            f"`{OGBN_NOTOPO_JSON}` |"
        ),
        (
            "| 2026-04-27 | ogbn-arxiv | test | eval_logit | True | True | pending | pending | "
            f"`{OGBN_TOPO_JSON}` |"
        ),
    ]
    for row in pending_rows:
        topo = "| True |" in row
        text = replace_or_insert_table_row(text, row, "ogbn-arxiv", topo)

    RESULTS_MD.write_text(text)


def kill_sample_gen(gpu: int) -> None:
    proc = subprocess.run(["pgrep", "-af", f"sample_gen.py start {gpu}"], text=True, capture_output=True)
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        pid = int(parts[0])
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)


def tmux_run(session: str, gpu: int, use_topo: bool, out_path: Path, json_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tmux", "kill-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = f"""cd {BASE} && source ~/.zshrc >/dev/null 2>&1 || true; conda activate /home/lingjie7/anaconda3/envs/dllm || conda activate dllm || true; echo '[start] ogbn-arxiv frozen {'topo' if use_topo else 'notopo'} eval at '$(date) > {out_path}; CUDA_VISIBLE_DEVICES={gpu} {PY} {BASE}/examples/tmdlm/eval_logit.py --exp ogbn_arxiv_llaga_frozen_{'topo' if use_topo else 'notopo'}_logit_labelon --model_name_or_path {MODEL} --dataset_name ogbn-arxiv --split test --batch_size 1 --max_seq_len 2048 --max_neighbors_per_hop 10 --max_hops 2 --use_topology_mask {str(use_topo)} --position_id_type sequential --prompt_format category_infill --prompt_layout target_first --max_answer_tokens 2 --include_neighbor_labels True --neighbor_label_format bracket --seed 42 --log_file {json_path} >> {out_path} 2>&1; rc=$?; echo '[eval-exit] rc='$rc' at '$(date) >> {out_path}; {PY} {SAMPLE_GEN} start {gpu} >> {out_path} 2>&1"""
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, cmd])


def main() -> None:
    print("[watcher] waiting for Cora and PubMed topo JSONL files", flush=True)
    while True:
        cora = read_last_json(CORA_JSON)
        pub = read_last_json(PUB_JSON)
        if cora and pub:
            break
        time.sleep(30)

    print(f"[watcher] Cora topo accuracy={cora['accuracy']}", flush=True)
    print(f"[watcher] PubMed topo accuracy={pub['accuracy']}", flush=True)
    update_results(cora, pub)
    print(f"[watcher] updated {RESULTS_MD}", flush=True)

    # Free the GPUs used by the completed frozen eval post-hooks.
    kill_sample_gen(3)
    kill_sample_gen(7)

    tmux_run(
        session="ogbn_arxiv_llaga_frozen_notopo_logit_gpu3",
        gpu=3,
        use_topo=False,
        out_path=OGBN_NOTOPO_OUT,
        json_path=OGBN_NOTOPO_JSON,
    )
    tmux_run(
        session="ogbn_arxiv_llaga_frozen_topo_logit_gpu7",
        gpu=7,
        use_topo=True,
        out_path=OGBN_TOPO_OUT,
        json_path=OGBN_TOPO_JSON,
    )
    print("[watcher] launched ogbn-arxiv no-topo on GPU3 and topo on GPU7", flush=True)


if __name__ == "__main__":
    main()
