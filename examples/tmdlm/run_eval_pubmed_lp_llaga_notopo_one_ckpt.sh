#!/usr/bin/env bash
# Single-checkpoint no-topo eval for PubMed LP on the official LLaGA test split.
#
# Usage:
#   sbatch --export=ALL,CKPT_DIR=<checkpoint-path> \
#     examples/tmdlm/run_eval_pubmed_lp_llaga_notopo_one_ckpt.sh

#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=pm-lp-notopo-1ckpt
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_notopo_1ckpt_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_notopo_1ckpt_%j.log

set -euo pipefail

: "${CKPT_DIR:?CKPT_DIR must be set}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_JSONL:-${LOG_DIR}/eval_pubmed_lp_llaga_notopo_allckpts.jsonl}"

EXP_TAG="pubmed_lp_llaga_notopo_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
  --exp "${EXP_TAG}" \
  --dataset_name pubmed \
  --model_name_or_path "${LLADA_PATH}" \
  --lora_path "${CKPT_DIR}" \
  --max_seq_len 4096 \
  --max_neighbors_per_hop 10 \
  --max_hops 2 \
  --use_topology_mask False \
  --position_id_type sequential \
  --max_samples 0 \
  --batch_size 8 \
  --log_file "${LOG_JSONL}"

echo "[eval] exit=$?"
echo "[eval] tail of jsonl:"
tail -n 1 "${LOG_JSONL}" || true
