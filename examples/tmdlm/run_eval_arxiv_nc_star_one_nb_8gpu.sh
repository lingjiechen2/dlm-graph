#!/usr/bin/env bash
# One ogbn-arxiv NC star-topology neighbor-budget eval on 8 GPUs.
# Submit with:
#   sbatch --export=ALL,NB=3 examples/tmdlm/run_eval_arxiv_nc_star_one_nb_8gpu.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --job-name=arxiv-nc-star-nb
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_star_nb%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_star_nb%j.log

set -euo pipefail

: "${NB:?Set NB, e.g. sbatch --export=ALL,NB=3 $0}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT/.helpers:$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

BASE_MODEL=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
CKPT_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-arxiv-nc-2hop-topo-r128-ep9-arxiv_nc_topo_20260524_0357_64gpu_9ep_r128_digit0pad_keep2/checkpoint-final
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
LOG_JSONL="${LOG_ROOT}/eval_nc_star_neighbor_sweep_20260526.jsonl"
mkdir -p "$LOG_ROOT"

EXP_TAG="arxiv_nc_star_nb${NB}_${SLURM_JOB_ID:-local}"
RANK_LOG_DIR="${LOG_ROOT}/arxiv_nc_star_nb${NB}_${SLURM_JOB_ID:-local}_torchrun"
mkdir -p "$RANK_LOG_DIR"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] job=${SLURM_JOB_ID:-?} node=$(hostname) nb=${NB}"
echo "[launch] ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
echo "[launch] rank_log_dir=${RANK_LOG_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  --log_dir "${RANK_LOG_DIR}" \
  --tee 3 \
  -- \
  examples/tmdlm/eval_logit.py \
    --exp "${EXP_TAG}" \
    --model_name_or_path "${BASE_MODEL}" \
    --lora_path "${CKPT_DIR}" \
    --dataset_name ogbn-arxiv \
    --split test \
    --max_samples 0 \
    --max_seq_len 4096 \
    --max_hops 2 \
    --max_neighbors_per_hop "${NB}" \
    --use_topology_mask True \
    --topology_mask_type star \
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

echo "[eval] exit=$? tail of jsonl:"
tail -n 1 "${LOG_JSONL}" || true
