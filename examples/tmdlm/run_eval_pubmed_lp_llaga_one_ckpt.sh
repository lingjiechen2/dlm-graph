#!/usr/bin/env bash
# Single-ckpt full eval for the LP-LLaGA eval pipeline on a single Cora LP SFT checkpoint.
# Runs eval_lp_llaga_split.py on 1× H200, on a 100-sample subset, to verify
# that the eval pipeline works end-to-end on the cluster after the recent
# data/weight migration to central locations.
#
# Usage: sbatch examples/tmdlm/run_eval_pubmed_lp_llaga_per_ckpt.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=pubmed-lp-eval-1ckpt
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_eval_1ckpt_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_eval_1ckpt_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

# Highest stable checkpoint available at submission time. The training job
# (1674656) finished while we were preparing this eval; using its highest
# numeric checkpoint (step 170, end of epoch 5 — functionally equivalent to
# checkpoint-final). Read-only access.


LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_DIR}/eval_pubmed_lp_llaga_per_ckpt.jsonl"

EXP_TAG="pubmed_lp_llaga_per_ckpt_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
  --exp "${EXP_TAG}" \
  --dataset_name pubmed \
  --model_name_or_path "${LLADA_PATH}" \
  --lora_path "${CKPT_DIR}" \
  --max_seq_len 4096 \
  --max_neighbors_per_hop 10 \
  --max_hops 2 \
  --use_topology_mask True \
  --position_id_type sequential \
  --max_samples 0 \
  --batch_size 4 \
  --log_file "${LOG_JSONL}"

echo "[eval] exit=$?"
echo "[eval] tail of jsonl:"
tail -n 1 "${LOG_JSONL}" || true
