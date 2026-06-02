#!/usr/bin/env bash
# Unified LP evaluation launcher for TM-DLM.
# Evaluates a single checkpoint on the LLaGA link-prediction test split.
#
# Required env vars:
#   DATASET    — cora | pubmed | ogbn-arxiv  (source/training dataset)
#   CKPT_DIR   — path to LoRA checkpoint directory
#
# Optional env vars:
#   TARGET_DATASET  defaults to DATASET (set for cross-dataset eval)
#   NGPU            8       GPUs to use (1 node)
#   MAX_SAMPLES     0       0 = full test split
#   TOPO            True    topology masking
#   NB              10      max_neighbors_per_hop
#   BATCH_SIZE      16
#   LOG_JSONL       auto-set based on job name
#
# In-domain example:
#   sbatch --export=ALL,DATASET=cora,CKPT_DIR=<path> \
#          examples/tmdlm/run_eval_lp.sh
#
# Cross-dataset example (Cora model → ogbn-arxiv):
#   sbatch --gres=gpu:8 \
#          --export=ALL,DATASET=cora,TARGET_DATASET=ogbn-arxiv,CKPT_DIR=<path> \
#          examples/tmdlm/run_eval_lp.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --job-name=tmdlm-eval-lp
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_lp_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_lp_%j.log

set -euo pipefail

: "${DATASET:?DATASET must be set (cora | pubmed | ogbn-arxiv)}"
: "${CKPT_DIR:?CKPT_DIR must be set}"

TARGET_DATASET="${TARGET_DATASET:-$DATASET}"
NGPU="${NGPU:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TOPO="${TOPO:-True}"
NB="${NB:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

NV_LIBS=$(ls -d "${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib" 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"

CROSS_TAG=$([ "$TARGET_DATASET" = "$DATASET" ] && echo "" || echo "_cross_${TARGET_DATASET}")
LOG_JSONL="${LOG_JSONL:-${LOG_DIR}/eval_${DATASET}_lp${CROSS_TAG}_${SLURM_JOB_ID:-local}.jsonl}"
EXP_TAG="${DATASET}_lp${CROSS_TAG}_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-local} gpus=${NGPU}"
echo "[launch] source=${DATASET} target=${TARGET_DATASET} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MASTER_PORT=$((29500 + RANDOM % 1000))

"$PY" -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
    --exp "${EXP_TAG}" \
    --dataset_name "${TARGET_DATASET}" \
    --model_name_or_path "${LLADA_PATH}" \
    --lora_path "${CKPT_DIR}" \
    --max_seq_len 4096 \
    --max_neighbors_per_hop "${NB}" \
    --max_hops 2 \
    --use_topology_mask "${TOPO}" \
    --position_id_type sequential \
    --max_samples "${MAX_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --log_file "${LOG_JSONL}"

echo "[eval] exit=$?"
tail -n 1 "${LOG_JSONL}" || true
