#!/usr/bin/env bash
# Evaluate partial Cora NC topo checkpoints on the Cora test split.
#
# Submit with:
#   sbatch --array=0-13%14 examples/tmdlm/run_eval_cora_nc_partial_ckpts_array.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --job-name=cora-nc-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_eval_partial_%A_%a.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_eval_partial_%A_%a.log

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
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

BASE_MODEL=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_ROOT"

CORA_ONLY=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-cora-nc-2hop-topo-mcdigit-d0-nonb-r64-ep10-cora_nc_topo_only_replicate_20260523_2219_mcdigit_d0_nonb
BALANCED=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-cora-pubmed-nc-2hop-topo-mcdigit-d0-bal-nonb-r64-ep10-cora_nc_topo_replicate_20260523_2109_mcdigit_d0_bal_nonb

TASKS=(
  "cora_only checkpoint-26  ${CORA_ONLY}/checkpoint-26"
  "cora_only checkpoint-52  ${CORA_ONLY}/checkpoint-52"
  "cora_only checkpoint-78  ${CORA_ONLY}/checkpoint-78"
  "cora_only checkpoint-104 ${CORA_ONLY}/checkpoint-104"
  "cora_only checkpoint-130 ${CORA_ONLY}/checkpoint-130"
  "cora_only checkpoint-156 ${CORA_ONLY}/checkpoint-156"
  "cora_only checkpoint-182 ${CORA_ONLY}/checkpoint-182"
  "cora_only checkpoint-208 ${CORA_ONLY}/checkpoint-208"
  "balanced  checkpoint-51  ${BALANCED}/checkpoint-51"
  "balanced  checkpoint-102 ${BALANCED}/checkpoint-102"
  "balanced  checkpoint-153 ${BALANCED}/checkpoint-153"
  "balanced  checkpoint-204 ${BALANCED}/checkpoint-204"
  "balanced  checkpoint-255 ${BALANCED}/checkpoint-255"
  "balanced  checkpoint-306 ${BALANCED}/checkpoint-306"
)

idx="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
if (( idx < 0 || idx >= ${#TASKS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}; have ${#TASKS[@]} tasks" >&2
  exit 2
fi

read -r group ckpt_name ckpt_dir <<< "${TASKS[$idx]}"
if [[ ! -f "${ckpt_dir}/adapter_config.json" ]]; then
  echo "Missing checkpoint adapter_config.json: ${ckpt_dir}" >&2
  exit 3
fi

jsonl="${LOG_ROOT}/eval_cora_nc_partial_${group}_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
exp="eval_cora_nc_${group}_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[launch] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[launch] group=${group} ckpt=${ckpt_name} ckpt_dir=${ckpt_dir}"
echo "[launch] jsonl=${jsonl}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" examples/tmdlm/eval_logit.py \
  --exp "$exp" \
  --model_name_or_path "$BASE_MODEL" \
  --lora_path "$ckpt_dir" \
  --dataset_name cora \
  --split test \
  --max_hops 2 \
  --max_neighbors_per_hop 10 \
  --max_seq_len 2048 \
  --use_topology_mask True \
  --position_id_type sequential \
  --prompt_format mc_digit \
  --answer_label_style digit0 \
  --max_answer_tokens 1 \
  --include_neighbor_labels False \
  --batch_size 8 \
  --log_file "$jsonl"

echo "[done] ${exp}"
