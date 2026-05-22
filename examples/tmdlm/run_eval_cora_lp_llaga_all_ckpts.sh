#!/usr/bin/env bash
# Full LLaGA-test eval (n=680) sweeping all checkpoints of the just-trained
# Cora LP run (LLaGA-split, 8 GPU, 5 epochs). One GPU sequential across the
# 11 ckpts (17, 34, 51, 68, 85, 102, 119, 136, 153, 170, final) ≈ 11 × ~5 min
# each ≈ 1 hour wall.
#
# Uses the dedicated dlm-graph conda env with PYTHONNOUSERSITE=1 so user-site
# doesn't shadow env packages.
#
# Usage: sbatch examples/tmdlm/run_eval_cora_lp_llaga_all_ckpts.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=cora-lp-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_lp_eval_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_lp_eval_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

# Conda's torch wheel ships its own CUDA/cuDNN/NCCL libs under nvidia/*/lib;
# add all of them so cuDNN (sdpa), cublas, etc. resolve at runtime.
# CUDNN_STATUS_NOT_INITIALIZED otherwise.
NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

# Run directory containing all checkpoints to eval. The SFT job 1674656 wrote
# to .models/ initially; mv to ~/model/dlm-graph/ is pending (task #4) and once
# done, update this path.
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/.models/tmdlm-llada-8b-cora-lp-2hop-r64-ep5-cora_lp_llaga_20260521_1751_8gpu_5ep}"

LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_DIR}/eval_cora_lp_llaga_allckpts.jsonl"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Sort checkpoint names numerically; "checkpoint-final" comes last.
mapfile -t CKPTS < <(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -V)

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "[error] no checkpoints under ${RUN_DIR}" >&2
  exit 2
fi

echo "[plan] ${#CKPTS[@]} checkpoints to evaluate:"
printf '  %s\n' "${CKPTS[@]}"

for CKPT_DIR in "${CKPTS[@]}"; do
  CKPT_NAME=$(basename "${CKPT_DIR}")
  EXP_TAG="cora_lp_llaga_${CKPT_NAME}_${SLURM_JOB_ID:-local}"
  echo
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] eval ${CKPT_NAME}"
  echo "============================================================"

  "$PY" "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
    --exp "${EXP_TAG}" \
    --dataset_name cora \
    --model_name_or_path "${LLADA_PATH}" \
    --lora_path "${CKPT_DIR}" \
    --max_seq_len 4096 \
    --max_neighbors_per_hop 10 \
    --max_hops 2 \
    --use_topology_mask True \
    --position_id_type sequential \
    --max_samples 0 \
    --batch_size 8 \
    --log_file "${LOG_JSONL}"
done

echo
echo "[done] all checkpoints evaluated. jsonl summary:"
tail -n ${#CKPTS[@]} "${LOG_JSONL}"
