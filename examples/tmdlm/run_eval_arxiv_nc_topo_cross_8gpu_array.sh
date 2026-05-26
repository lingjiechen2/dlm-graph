#!/usr/bin/env bash
# Cross-dataset NC eval for arxiv-trained topology-masked checkpoints.
#
# Evaluates checkpoint-5760, checkpoint-6399, and checkpoint-final on:
#   cora, pubmed, ogbn-arxiv
#
# Submit:
#   sbatch --array=0-8%9 examples/tmdlm/run_eval_arxiv_nc_topo_cross_8gpu_array.sh
#
#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --job-name=arxiv-nc-cross-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_cross_eval_%A_%a.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_cross_eval_%A_%a.log

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

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

BASE_MODEL=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
RUN_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-arxiv-nc-2hop-topo-r128-ep9-arxiv_nc_topo_20260524_0357_64gpu_9ep_r128_digit0pad_keep2
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_ROOT"

CKPTS=(checkpoint-5760 checkpoint-6399 checkpoint-final)
DATASETS=(cora pubmed ogbn-arxiv)

idx="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
num_ckpts="${#CKPTS[@]}"
num_tasks=$(( num_ckpts * ${#DATASETS[@]} ))
if (( idx < 0 || idx >= num_tasks )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}; have ${num_tasks} tasks" >&2
  exit 2
fi

ckpt_idx=$(( idx / ${#DATASETS[@]} ))
dataset_idx=$(( idx % ${#DATASETS[@]} ))
ckpt_name="${CKPTS[$ckpt_idx]}"
dataset="${DATASETS[$dataset_idx]}"
ckpt_dir="${RUN_DIR}/${ckpt_name}"

if [[ ! -f "${ckpt_dir}/adapter_config.json" ]]; then
  echo "Missing checkpoint adapter_config.json: ${ckpt_dir}" >&2
  exit 3
fi

case "$dataset" in
  cora|pubmed)
    answer_label_style=digit0
    max_answer_tokens=1
    ;;
  ogbn-arxiv)
    answer_label_style=digit0_pad
    max_answer_tokens=2
    ;;
  *)
    echo "Unsupported dataset: ${dataset}" >&2
    exit 4
    ;;
esac

safe_dataset="${dataset//[^A-Za-z0-9]/_}"
jsonl="${LOG_ROOT}/eval_arxiv_nc_topo_cross_${safe_dataset}_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
exp="arxiv_nc_topo_${ckpt_name}_to_${safe_dataset}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rank_log_dir="${LOG_ROOT}/arxiv_nc_cross_${safe_dataset}_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_torchrun"
mkdir -p "$rank_log_dir"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[launch] source=arxiv-nc-topo ckpt=${ckpt_name} target=${dataset}"
echo "[launch] ckpt_dir=${ckpt_dir}"
echo "[launch] answer_label_style=${answer_label_style} max_answer_tokens=${max_answer_tokens}"
echo "[launch] jsonl=${jsonl}"
echo "[launch] rank_log_dir=${rank_log_dir}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  --log_dir "$rank_log_dir" \
  --tee 3 \
  -- \
  examples/tmdlm/eval_logit.py \
    --exp "$exp" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$ckpt_dir" \
    --dataset_name "$dataset" \
    --split test \
    --max_samples 0 \
    --max_hops 2 \
    --max_neighbors_per_hop 10 \
    --max_seq_len 4096 \
    --use_topology_mask True \
    --position_id_type sequential \
    --prompt_format mc_digit \
    --answer_label_style "$answer_label_style" \
    --max_answer_tokens "$max_answer_tokens" \
    --include_neighbor_labels False \
    --batch_size 2 \
    --log_file "$jsonl"

echo "[done] tail:"
tail -n 1 "$jsonl" || true
