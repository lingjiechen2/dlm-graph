"""
Monitor GPU memory usage and utilization, and launch sample_gen workers on idle GPUs.

Run:
    source ~/.zshrc
    conda activate /home/lingjie7/anaconda3/envs/dllm
    /home/lingjie7/anaconda3/envs/dllm/bin/python /home/lingjie7/auto-research/projects/dlm-graph/scripts/data_preprocessing.py

Examples:
    /home/lingjie7/anaconda3/envs/dllm/bin/python /home/lingjie7/auto-research/projects/dlm-graph/scripts/data_preprocessing.py --gpus 0 1 2 3
    /home/lingjie7/anaconda3/envs/dllm/bin/python /home/lingjie7/auto-research/projects/dlm-graph/scripts/data_preprocessing.py --threshold-percent 5 --poll-seconds 5
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


PYTHON_EXECUTABLE = "/home/lingjie7/anaconda3/envs/dllm/bin/python"
SAMPLE_GEN_PATH = Path("/home/lingjie7/sample_gen.py")


def query_gpu_stats() -> dict[int, tuple[int, int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    usage: dict[int, tuple[int, int, int]] = {}
    for line in result.stdout.strip().splitlines():
        gpu_str, used_str, total_str, util_str = [
            part.strip() for part in line.split(",")
        ]
        usage[int(gpu_str)] = (int(used_str), int(total_str), int(util_str))
    return usage


def launch_sample_gen(gpu_id: int) -> subprocess.CompletedProcess[str]:
    cmd = [PYTHON_EXECUTABLE, str(SAMPLE_GEN_PATH), "start", str(gpu_id)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def print_snapshot(
    usage: dict[int, tuple[int, int, int]],
    target_gpus: list[int],
) -> None:
    print("[snapshot]", flush=True)
    for gpu_id in target_gpus:
        if gpu_id not in usage:
            print(f"  gpu={gpu_id} missing", flush=True)
            continue
        used_mb, total_mb, util_pct = usage[gpu_id]
        used_pct = 100.0 * used_mb / total_mb if total_mb else 100.0
        print(
            f"  gpu={gpu_id} mem={used_mb}/{total_mb}MB ({used_pct:.2f}%) util={util_pct}%",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch sample_gen.py on GPUs whose memory usage is below a threshold."
    )
    parser.add_argument(
        "--gpus",
        nargs="*",
        type=int,
        default=None,
        help="Specific GPU indices to monitor. Default: all visible GPUs.",
    )
    parser.add_argument(
        "--threshold-percent",
        type=float,
        default=5.0,
        help="Launch when used memory / total memory is below this percentage.",
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=5.0,
        help="Launch when GPU utilization is below this percentage.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every poll instead of only printing state changes.",
    )
    args = parser.parse_args()

    print(
        f"[monitor] sample_gen={SAMPLE_GEN_PATH} mem_threshold={args.threshold_percent:.2f}% util_threshold={args.util_threshold:.2f}% poll={args.poll_seconds:.1f}s",
        flush=True,
    )
    last_state: dict[int, str] = {}
    last_stats: dict[int, tuple[int, int]] = {}
    usage = query_gpu_stats()
    target_gpus = (
        args.gpus if args.gpus is not None and len(args.gpus) > 0 else sorted(usage)
    )
    print_snapshot(usage, target_gpus)

    while True:
        usage = query_gpu_stats()
        target_gpus = (
            args.gpus if args.gpus is not None and len(args.gpus) > 0 else sorted(usage)
        )
        for gpu_id in target_gpus:
            if gpu_id not in usage:
                state = "missing"
                if args.verbose or last_state.get(gpu_id) != state:
                    print(f"[skip] gpu={gpu_id} not found in nvidia-smi", flush=True)
                last_state[gpu_id] = state
                continue
            used_mb, total_mb, util_pct = usage[gpu_id]
            used_pct = 100.0 * used_mb / total_mb if total_mb else 100.0
            now_stats = (round(used_pct), util_pct)
            prev_stats = last_stats.get(gpu_id)
            changed_enough = (
                prev_stats is None
                or abs(now_stats[0] - prev_stats[0]) >= 3
                or abs(now_stats[1] - prev_stats[1]) >= 10
            )
            if used_pct < args.threshold_percent and util_pct < args.util_threshold:
                state = "idle"
                result = launch_sample_gen(gpu_id)
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                if (
                    args.verbose
                    or last_state.get(gpu_id) != state
                    or changed_enough
                    or stdout
                ):
                    if stdout:
                        print(
                            f"[launch] gpu={gpu_id} mem={used_mb}/{total_mb}MB ({used_pct:.2f}%) util={util_pct}% {stdout}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[launch] gpu={gpu_id} mem={used_mb}/{total_mb}MB ({used_pct:.2f}%) util={util_pct}% return_code={result.returncode}",
                            flush=True,
                        )
                if stderr:
                    print(f"[stderr] gpu={gpu_id} {stderr}", flush=True)
                last_state[gpu_id] = state
            else:
                state = "busy"
                if args.verbose or last_state.get(gpu_id) != state or changed_enough:
                    print(
                        f"[busy] gpu={gpu_id} mem={used_mb}/{total_mb}MB ({used_pct:.2f}%) util={util_pct}%",
                        flush=True,
                    )
                last_state[gpu_id] = state
            last_stats[gpu_id] = now_stats
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
