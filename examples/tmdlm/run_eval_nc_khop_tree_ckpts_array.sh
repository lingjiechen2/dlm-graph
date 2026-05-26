#!/usr/bin/env bash
# Evaluate khop-tree Cora/PubMed NC checkpoints on test splits.
#
# Submit with:
#   sbatch --array=0-11%6 examples/tmdlm/run_eval_nc_khop_tree_ckpts_array.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=khop-nc-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/khop_nc_eval_%A_%a.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/khop_nc_eval_%A_%a.log

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
RUN_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-cora-pubmed-nc-2hop-khoptree-mcdigit-d0-bal-nonb-r64-ep10-cora_nc_khop_tree_8gpu_20260523_2245_mcdigit_d0_bal_nonb
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_ROOT"

CKPTS=(7 14 21 28 35 42)
DATASETS=(cora pubmed)

idx="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
num_ckpts="${#CKPTS[@]}"
num_tasks=$(( num_ckpts * ${#DATASETS[@]} ))
if (( idx < 0 || idx >= num_tasks )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}; have ${num_tasks} tasks" >&2
  exit 2
fi

dataset_idx=$(( idx / num_ckpts ))
ckpt_idx=$(( idx % num_ckpts ))
dataset="${DATASETS[$dataset_idx]}"
step="${CKPTS[$ckpt_idx]}"
ckpt_dir="${RUN_DIR}/checkpoint-${step}"

if [[ ! -f "${ckpt_dir}/adapter_config.json" ]]; then
  echo "Missing checkpoint adapter_config.json: ${ckpt_dir}" >&2
  exit 3
fi

jsonl="${LOG_ROOT}/eval_nc_khop_tree_ckpts.jsonl"
exp="eval_${dataset}_nc_khop_tree_checkpoint_${step}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[launch] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[launch] dataset=${dataset} checkpoint=checkpoint-${step}"
echo "[launch] ckpt_dir=${ckpt_dir}"
echo "[launch] jsonl=${jsonl}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" examples/tmdlm/eval_logit.py \
  --exp "$exp" \
  --model_name_or_path "$BASE_MODEL" \
  --lora_path "$ckpt_dir" \
  --dataset_name "$dataset" \
  --split test \
  --max_hops 2 \
  --max_neighbors_per_hop 10 \
  --max_seq_len 2048 \
  --use_topology_mask True \
  --topology_mask_type khop_tree \
  --position_id_type sequential \
  --prompt_format mc_digit \
  --answer_label_style digit0 \
  --max_answer_tokens 1 \
  --include_neighbor_labels False \
  --batch_size 8 \
  --log_file "$jsonl"

echo "[done] ${exp}"
tail -n 3 "$jsonl" || true
