#!/usr/bin/env bash
# Cora-only NC topo checkpoints evaluated on ogbn-arxiv.
#
# Submit:
#   sbatch --array=0-1%2 examples/tmdlm/run_eval_cora_nc_topo_to_arxiv_last2_8gpu_array.sh
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
#SBATCH --job-name=cora2arxiv-nc-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_topo_to_arxiv_%A_%a.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_topo_to_arxiv_%A_%a.log

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
RUN_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-cora-nc-2hop-topo-mcdigit-d0-nonb-r64-ep10-cora_nc_topo_only_replicate_20260523_2219_mcdigit_d0_nonb
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_ROOT"

CKPTS=(checkpoint-182 checkpoint-208)
idx="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"
if (( idx < 0 || idx >= ${#CKPTS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${idx}; have ${#CKPTS[@]} tasks" >&2
  exit 2
fi

ckpt_name="${CKPTS[$idx]}"
ckpt_dir="${RUN_DIR}/${ckpt_name}"
if [[ ! -f "${ckpt_dir}/adapter_config.json" ]]; then
  echo "Missing checkpoint adapter_config.json: ${ckpt_dir}" >&2
  exit 3
fi

jsonl="${LOG_ROOT}/eval_cora_nc_topo_to_arxiv_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"
exp="cora_nc_topo_${ckpt_name}_to_arxiv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
rank_log_dir="${LOG_ROOT}/cora_nc_topo_to_arxiv_${ckpt_name}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_torchrun"
mkdir -p "$rank_log_dir"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=$(hostname)"
echo "[launch] source=cora-nc-topo ckpt=${ckpt_name} target=ogbn-arxiv"
echo "[launch] ckpt_dir=${ckpt_dir}"
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
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 0 \
    --max_hops 2 \
    --max_neighbors_per_hop 10 \
    --max_seq_len 4096 \
    --use_topology_mask True \
    --position_id_type sequential \
    --prompt_format mc_digit \
    --answer_label_style digit0_pad \
    --max_answer_tokens 2 \
    --include_neighbor_labels False \
    --batch_size 2 \
    --log_file "$jsonl"

echo "[done] tail:"
tail -n 1 "$jsonl" || true
