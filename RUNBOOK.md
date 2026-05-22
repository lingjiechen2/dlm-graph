# dlm-graph Cluster Runbook (fs-mbz Slurm)

Operational notes for running dlm-graph SFT and eval on the `fs-mbz-login-big-001` Slurm cluster. Project-specific; for global storage conventions see `~/.claude/CLAUDE.md`.

## Paths

| Asset | Location |
|---|---|
| Repo | `~/dlm-graph/` |
| Conda env | `~/miniconda3/envs/dlm-graph/` |
| LLaDA-8B-Instruct weights | `~/model/huggingface/GSAI-ML/LLaDA-8B-Instruct/` |
| LLaGA datasets | `~/dataset/dlm-graph/llaga/{cora,pubmed,ogbn-arxiv}/` |
| Trained adapters / outputs | `~/model/dlm-graph/<run-tag>/` |
| Slurm + run logs | `~/model/dlm-graph/logs/` |

Never write models or datasets inside the repo (no `.models/`, no `.datasets/`).

## Required env (set in every sbatch)

```bash
CONDA_ENV=~/miniconda3/envs/dlm-graph
PY=$CONDA_ENV/bin/python
export PYTHONNOUSERSITE=1                                # block ~/.local/ from shadowing the env
export LLAGA_DATA_ROOT=~/dataset/dlm-graph/llaga
export PYTHONPATH=~/dlm-graph:${PYTHONPATH:-}

# torch's bundled CUDA / cuDNN / NCCL / cublas / etc. live under nvidia/*/lib.
# The dynamic linker doesn't see them by default → cuDNN init fails on the first
# sdpa call (CUDNN_STATUS_NOT_INITIALIZED) and multi-GPU NCCL init fails
# (ncclDevCommCreate). Add them all:
NV_LIBS=$(ls -d $CONDA_ENV/lib/python3.10/site-packages/nvidia/*/lib | paste -sd: -)
export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"
```

## Slurm

Account is `arch`. Allowed QoS: `arch`, `cpuonly`, `lowprio`. Non-default partitions need explicit matching `--qos`.

**QoS choice**:
- `--partition=main --qos=arch` — interactive / iteration / smoke tests. Usually grabs nodes immediately even for 3-node multi-node jobs (observed: 0 wait on 3×8 GPU).
- `--partition=lowprio --qos=lowprio` — long batch jobs (multi-hour SFT). Lower priority, scheduler backfills. Multi-node lowprio can wait 10+ min for the same allocation that `arch` gets instantly. Use when you don't want to burn `arch` fairshare on a long run.
- `--partition=cpuonly --qos=cpuonly` — CPU-only work (env setup, dataset prep, eval prompts without GPU). MaxWall is 4 days.

**Walltime**: `arch` / `lowprio` / `highprio` QoS have **no MaxWall** (`infinite`). Don't tightly bound `--time=` to your estimated wall — set generously so transient slowdowns (FS contention, longer-than-expected build, etc.) don't kill the job. Defaults used in this repo: GPU SFT/eval `--time=12:00:00`, GPU smoke/test `--time=02:00:00`, CPU jobs `--time=04:00:00`.

Templates:

```bash
# 8-GPU single-node DDP SFT
#SBATCH --partition=lowprio --qos=lowprio --account=arch
#SBATCH --gres=gpu:8 --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=64 --mem=512G --time=04:00:00

# 1-GPU eval / single-job
#SBATCH --partition=lowprio --qos=lowprio --account=arch
#SBATCH --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=03:00:00
```

GPU live-watch from login node (while job is running):

```bash
srun --jobid=<JID> --overlap --pty nvitop                    # full TUI
srun --jobid=<JID> --overlap gpustat -i 2                    # 2s rolling
srun --jobid=<JID> --overlap nvidia-smi                      # one-shot
```

## Parallel eval: one ckpt per GPU

Eval is embarrassingly parallel across checkpoints. The default sweep (`run_eval_cora_lp_llaga_all_ckpts.sh`) walks all ckpts sequentially on 1 GPU — easy to manage but slow. With 11 ckpts × ~1:40 each that's ~18 min of wall time.

Better when iterating: submit one sbatch per ckpt, each on its own GPU. ~1:40 wall total instead of 18 min.

Pattern A (driver script + per-ckpt sbatch):

```bash
RUN_DIR=~/model/dlm-graph/<run-tag>
for CKPT in $RUN_DIR/checkpoint-*; do
  sbatch --export=ALL,CKPT_DIR="$CKPT" examples/tmdlm/run_eval_cora_lp_llaga_smoke.sh
done
```

The eval sbatch reads `${CKPT_DIR:-<default>}`, so `--export=ALL,CKPT_DIR=...` overrides it per submission. Each job grabs 1 H200 from lowprio.

Pattern B (Slurm job array):

```bash
mapfile -t CKPTS < <(ls -d $RUN_DIR/checkpoint-*)
# in the sbatch:
#SBATCH --array=0-10
CKPT_LIST=( "${CKPTS[@]}" )                          # hardcode after `mapfile` above the sbatch
CKPT_DIR=${CKPT_LIST[$SLURM_ARRAY_TASK_ID]}
```

Job arrays share one `sbatch` submission but each task gets independent GPU resources from the scheduler. Logs split per-task automatically (`%a` in `--output`).

When NOT to parallelize: if the cluster is heavily loaded and `--qos=lowprio` queues, a single sequential job often runs sooner than 11 parallel jobs all waiting. Check `sinfo -p lowprio --states=idle` first.

## Known pitfalls

1. **Half-installed cuDNN**: parallel pip installs can truncate `libcudnn_engines_precompiled.so.9` (saw 245 MB vs the correct 515 MB after one collision on 2026-05-21). Symptom: cuDNN init returns NOT_INITIALIZED even though LD_LIBRARY_PATH is correct. Check the file size; if too small, force-reinstall: `pip install --force-reinstall --no-deps nvidia-cudnn-cu12`.

2. **NCCL version mismatch**: pip metadata can disagree with the on-disk `libnccl.so.2` (saw `nvidia-nccl-cu12 2.28.9` per pip but `NCCL version 2.29.7+cuda13.2` in the actual .so file). torch 2.11+cu128 expects 2.28.9+cuda12.9. Symptom: multi-GPU init dies with `DistBackendError: NCCL error ... unhandled cuda error`. Verify with `strings .../libnccl.so.2 | grep 'NCCL version'`; force-reinstall to match: `pip install --force-reinstall --no-deps nvidia-nccl-cu12==2.28.9`.

2. **User-site shadows conda env**: `~/.local/lib/python3.10/site-packages/` is on `sys.path` BEFORE the conda env's site-packages by default. Always set `PYTHONNOUSERSITE=1` when launching from the conda env, or risk loading a stale package from user-site.

3. **`rm -rf <glob>` on dirs containing dist-info**: unmatched globs become literal args; `-` prefixes get parsed as options by `rm`. Always `ls <glob>` first, and prefer `pip uninstall` for pip leftovers.

4. **`processed_data.pt` requires `weights_only=False`** (PyG `Data` object). `_common.py:65` was patched on 2026-05-21 to pass it; the eval script already had it.

5. **Multi-GPU LLaDA needs the full nvidia/*/lib on LD_LIBRARY_PATH**, not just NCCL. cuDNN, cublas, etc. are loaded lazily by torch at the first forward.

## Cross-references

- Global storage conventions: `~/.claude/CLAUDE.md`
- LP migration TODO: `~/dlm-graph/TODO_LP.md`
- Experiment log (NC): `~/dlm-graph/examples/tmdlm/experiment_log.md`
- Detailed results table: `~/dlm-graph/examples/tmdlm/results.md`
