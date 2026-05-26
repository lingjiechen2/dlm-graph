#!/usr/bin/env bash
# Single-checkpoint topo cross-dataset LP eval for ogbn-arxiv targets.
#
# Usage:
#   sbatch --export=ALL,SOURCE_DATASET=cora,TARGET_DATASET=ogbn-arxiv,CKPT_DIR=<path> \
#     examples/tmdlm/run_eval_lp_cross_topo_one_ckpt_8gpu.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --job-name=lp-cross-topo-8gpu
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/lp_cross_topo_8gpu_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/lp_cross_topo_8gpu_%j.log

set -euo pipefail

: "${SOURCE_DATASET:?SOURCE_DATASET must be set}"
: "${TARGET_DATASET:?TARGET_DATASET must be set}"
: "${CKPT_DIR:?CKPT_DIR must be set}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT/.helpers:$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_JSONL:-${LOG_DIR}/eval_lp_cross_dataset_topo_final.jsonl}"

EXP_TAG="lp_cross_topo_${SOURCE_DATASET}_to_${TARGET_DATASET}_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[launch] source=${SOURCE_DATASET} target=${TARGET_DATASET} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MASTER_PORT=$((29500 + RANDOM % 1000))

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
    --exp "${EXP_TAG}" \
    --dataset_name "${TARGET_DATASET}" \
    --model_name_or_path "${LLADA_PATH}" \
    --lora_path "${CKPT_DIR}" \
    --max_seq_len 4096 \
    --max_neighbors_per_hop 10 \
    --max_hops 2 \
    --use_topology_mask True \
    --position_id_type sequential \
    --max_samples 0 \
    --batch_size 16 \
    --log_file "${LOG_JSONL}"

echo "[eval] exit=$?"
tail -n 1 "${LOG_JSONL}" || true
