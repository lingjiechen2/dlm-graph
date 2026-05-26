#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-300}"
OUT="${OUT:-/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/monitor_nc_notopo_1682977_1682978_1682979.log}"

python_bin="${PYTHON_BIN:-python3}"

while true; do
  {
    date -u '+# %Y-%m-%d %H:%M:%S UTC'
    "$python_bin" - <<'PY'
import os
import re
import subprocess

jobs = [
    {
        "dataset": "cora",
        "job_id": "1682977",
        "target_epoch": 10.0,
        "log": "/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_notopo_keep2_1682977.log",
    },
    {
        "dataset": "pubmed",
        "job_id": "1682978",
        "target_epoch": 10.0,
        "log": "/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_nc_notopo_keep2_1682978.log",
    },
    {
        "dataset": "ogbn-arxiv",
        "job_id": "1682979",
        "target_epoch": 9.0,
        "log": "/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_notopo_64gpu_keep2_1682979.log",
    },
]

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ""

def elapsed_seconds(elapsed):
    if not elapsed:
        return 0
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-", 1)
        days = int(d)
    parts = [int(p) for p in elapsed.split(":")]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, 0, parts[0]
    return days * 86400 + h * 3600 + m * 60 + s

def fmt_seconds(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"

def parse_log(path):
    text = open(path, errors="replace").read() if os.path.exists(path) else ""
    out = re.search(r"output=([^\s]+)", text)
    losses = re.findall(r"\{'loss': ([^,]+), 'grad_norm': ([^,]+), 'learning_rate': ([^,]+), 'epoch': ([^}]+)\}", text)
    errors = len(re.findall(r"Traceback|\bERROR\b|Exception", text))
    run_dir = out.group(1) if out else ""
    ckpts = []
    if run_dir and os.path.isdir(run_dir):
        ckpts = sorted([d for d in os.listdir(run_dir) if d.startswith("checkpoint-")])
    if losses:
        loss, _grad, _lr, epoch = losses[-1]
        return run_dir, float(loss), float(epoch), ckpts, errors
    return run_dir, None, 0.0, ckpts, errors

squeue = run(["squeue", "-h", "-j", ",".join(j["job_id"] for j in jobs), "-o", "%i|%T|%M|%R"])
sq = {}
for line in squeue.splitlines():
    parts = line.split("|", 3)
    if len(parts) == 4:
        sq[parts[0].strip()] = {
            "state": parts[1].strip(),
            "elapsed": parts[2].strip(),
            "reason": parts[3].strip(),
        }

print("| Dataset | Job | State | Elapsed | Epoch | Progress | Latest loss | ETA | Checkpoints | Errors |")
print("|---|---:|---|---:|---:|---:|---:|---:|---|---:|")
for job in jobs:
    info = sq.get(job["job_id"], {"state": "not-in-squeue", "elapsed": "", "reason": ""})
    run_dir, loss, epoch, ckpts, errors = parse_log(job["log"])
    elapsed = elapsed_seconds(info["elapsed"])
    progress = epoch / job["target_epoch"] if job["target_epoch"] else 0.0
    eta = None
    if elapsed > 0 and progress > 0:
        eta = elapsed * (1.0 - progress) / progress
    loss_s = "none" if loss is None else f"{loss:.4f}"
    ckpt_s = "none" if not ckpts else " ".join(ckpts[-2:])
    print(
        f"| {job['dataset']} | {job['job_id']} | {info['state']} | {info['elapsed'] or 'n/a'} | "
        f"{epoch:.4f}/{job['target_epoch']:.0f} | {progress*100:.1f}% | {loss_s} | "
        f"{fmt_seconds(eta)} | {ckpt_s} | {errors} |"
    )
print()
PY
  } >> "$OUT"
  sleep "$INTERVAL"
done
