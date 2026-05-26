#!/usr/bin/env bash
# Full LLaGA-test eval for the ogbn-arxiv no-topo LP SFT run.
#
# Usage:
#   sbatch examples/tmdlm/run_eval_arxiv_lp_llaga_notopo_all_ckpts_8gpu.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=arxiv-lp-notopo-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_notopo_eval_8gpu_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_notopo_eval_8gpu_%j.log

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
RUN_DIR="${RUN_DIR:-/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-arxiv-lp-2hop-notopo-r64-ep5-arxiv_lp_llaga_notopo_20260523_0853_32gpu_5ep}"

LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"
LOG_JSONL="${LOG_JSONL:-${LOG_DIR}/eval_arxiv_lp_llaga_notopo_allckpts.jsonl}"

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-?}"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mapfile -t CKPTS < <(ls -d "${RUN_DIR}"/checkpoint-* 2>/dev/null | sort -V)

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "[error] no checkpoints under ${RUN_DIR}" >&2
  exit 2
fi

echo "[plan] ${#CKPTS[@]} checkpoints to evaluate:"
printf '  %s\n' "${CKPTS[@]}"

for CKPT_DIR in "${CKPTS[@]}"; do
  CKPT_NAME=$(basename "${CKPT_DIR}")
  EXP_TAG="arxiv_lp_llaga_notopo_${CKPT_NAME}_${SLURM_JOB_ID:-local}"
  MASTER_PORT=$((29500 + RANDOM % 1000))
  echo
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] eval ${CKPT_NAME}"
  echo "============================================================"

  "$PY" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    -- \
    "${REPO_ROOT}/examples/tmdlm/eval_lp_llaga_split.py" \
      --exp "${EXP_TAG}" \
      --dataset_name ogbn-arxiv \
      --model_name_or_path "${LLADA_PATH}" \
      --lora_path "${CKPT_DIR}" \
      --max_seq_len 4096 \
      --max_neighbors_per_hop 10 \
      --max_hops 2 \
      --use_topology_mask False \
      --position_id_type sequential \
      --max_samples 0 \
      --batch_size 16 \
      --log_file "${LOG_JSONL}"
done

echo
echo "[done] all checkpoints evaluated. jsonl summary:"
tail -n ${#CKPTS[@]} "${LOG_JSONL}"
