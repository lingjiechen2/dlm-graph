#!/usr/bin/env bash
# Frozen LLaDA-8B-Instruct LP eval on LLaGA ogbn-arxiv test split.
#
# Usage: sbatch examples/tmdlm/run_eval_frozen_llada_lp_llaga_arxiv_4gpu.sh
#
#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=06:00:00
#SBATCH --job-name=frozen-lp-arxiv
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/frozen_lp_arxiv_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/frozen_lp_arxiv_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT/.helpers:$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_DIR}/eval_frozen_llada_lp_llaga.jsonl"
EXP_TAG="frozen_llada_lp_llaga_ogbn_arxiv_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MASTER_PORT=$((29500 + RANDOM % 1000))

"$PY" -m torch.distributed.run \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
    --exp "${EXP_TAG}" \
    --dataset_name ogbn-arxiv \
    --model_name_or_path "${LLADA_PATH}" \
    --max_seq_len 4096 \
    --max_neighbors_per_hop 10 \
    --max_hops 2 \
    --use_topology_mask True \
    --position_id_type sequential \
    --max_samples 0 \
    --batch_size 16 \
    --log_file "${LOG_JSONL}"

"$PY" "${REPO_ROOT}/scripts/update_frozen_llada_lp_results.py" \
  --log-file "${LOG_JSONL}" \
  --results-md "${REPO_ROOT}/results.md" \
  --detailed-md "${REPO_ROOT}/results/current_results_detailed.md"

echo "[done] latest frozen LP JSONL row:"
tail -n 1 "${LOG_JSONL}" || true
