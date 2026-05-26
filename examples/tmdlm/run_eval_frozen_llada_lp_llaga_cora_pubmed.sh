#!/usr/bin/env bash
# Frozen LLaDA-8B-Instruct LP eval on LLaGA Cora and PubMed test splits.
#
# Usage: sbatch examples/tmdlm/run_eval_frozen_llada_lp_llaga_cora_pubmed.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --job-name=frozen-lp-cora-pm
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/frozen_lp_cora_pubmed_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/frozen_lp_cora_pubmed_%j.log

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

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_DIR}/eval_frozen_llada_lp_llaga.jsonl"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for dataset in cora pubmed; do
  exp="frozen_llada_lp_llaga_${dataset}_${SLURM_JOB_ID:-local}"
  echo "[$(date +%H:%M:%S)] eval ${dataset}"
  "$PY" "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
    --exp "${exp}" \
    --dataset_name "${dataset}" \
    --model_name_or_path "${LLADA_PATH}" \
    --max_seq_len 4096 \
    --max_neighbors_per_hop 10 \
    --max_hops 2 \
    --use_topology_mask True \
    --position_id_type sequential \
    --max_samples 0 \
    --batch_size 4 \
    --log_file "${LOG_JSONL}"
  "$PY" "${REPO_ROOT}/scripts/update_frozen_llada_lp_results.py" \
    --log-file "${LOG_JSONL}" \
    --results-md "${REPO_ROOT}/results.md" \
    --detailed-md "${REPO_ROOT}/results/current_results_detailed.md"
done

echo "[done] latest frozen LP JSONL rows:"
tail -n 3 "${LOG_JSONL}" || true
