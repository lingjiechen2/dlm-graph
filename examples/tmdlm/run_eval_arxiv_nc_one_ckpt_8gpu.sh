#!/usr/bin/env bash
# Per-ckpt arxiv NC eval on 8 GPUs (1 node), DDP-sharded test set.
# Each rank evaluates 1/8 of the ogbn-arxiv test split (~48k samples → ~6k/rank);
# per-class counts are all_reduced before rank 0 writes the jsonl.
# Recipe matches §23 (`arxiv_20260514_fulltrain_r128_3ep` / 9-ep replicate):
# digit0_pad + max_answer_tokens=2, full ogbn-arxiv train (no cap, no boost).
# Calibration still on (computes both raw + calibrated) but with uniform
# train spec — §22/§23 show cal < raw without boost, but we keep both metrics.
#
# Pass CKPT_DIR via --export=ALL,CKPT_DIR=... ; the script errors out (set -u)
# if not provided.
#
# Usage:
#   sbatch --export=ALL,CKPT_DIR=<...> examples/tmdlm/run_eval_arxiv_nc_one_ckpt_8gpu.sh
#
#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --job-name=arxiv-nc-eval-8gpu
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_eval_8gpu_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_eval_8gpu_%j.log

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
LOG_JSONL="${LOG_DIR}/eval_arxiv_nc_per_ckpt.jsonl"

EXP_TAG="arxiv_nc_per_ckpt_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MASTER_PORT=$((29500 + RANDOM % 1000))

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/eval_logit.py" \
    --exp "${EXP_TAG}" \
    --model_name_or_path "${LLADA_PATH}" \
    --lora_path "${CKPT_DIR}" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 0 \
    --max_seq_len 4096 \
    --max_hops 2 \
    --max_neighbors_per_hop 10 \
    --use_topology_mask True \
    --prompt_format mc_digit \
    --answer_label_style digit0_pad \
    --max_answer_tokens 2 \
    --include_neighbor_labels False \
    --position_id_type sequential \
    --batch_size 4 \
    --apply_class_prior_calibration True \
    --train_resample_strategy none \
    --train_boost_spec "" \
    --train_max_train_samples 0 \
    --log_file "${LOG_JSONL}"

echo "[eval] exit=$?  tail of jsonl:"
tail -n 1 "${LOG_JSONL}" || true
