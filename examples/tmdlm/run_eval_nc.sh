#!/usr/bin/env bash
# Unified NC evaluation launcher for TM-DLM.
# Evaluates a single checkpoint on the node-classification test split.
#
# Required env vars:
#   DATASET    — cora | pubmed | ogbn-arxiv  (training/source dataset)
#   CKPT_DIR   — path to LoRA checkpoint directory
#
# Optional env vars:
#   TARGET_DATASET  defaults to DATASET (set for cross-dataset eval)
#   NGPU            8       GPUs to use (1 node)
#   MAX_SAMPLES     0       0 = full test split
#   TOPO            True    topology masking
#   NB              10      max_neighbors_per_hop
#   LOG_JSONL       auto-set based on job name
#
# In-domain example:
#   sbatch --export=ALL,DATASET=ogbn-arxiv,CKPT_DIR=<path> \
#          examples/tmdlm/run_eval_nc.sh
#
# Cross-dataset example (PubMed model → Cora):
#   sbatch --export=ALL,DATASET=pubmed,TARGET_DATASET=cora,CKPT_DIR=<path> \
#          examples/tmdlm/run_eval_nc.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --job-name=tmdlm-eval-nc
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_nc_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/eval_nc_%j.log

set -euo pipefail

: "${DATASET:?DATASET must be set (cora | pubmed | ogbn-arxiv)}"
: "${CKPT_DIR:?CKPT_DIR must be set}"

TARGET_DATASET="${TARGET_DATASET:-$DATASET}"
NGPU="${NGPU:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TOPO="${TOPO:-True}"
NB="${NB:-10}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

NV_LIBS=$(ls -d "${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib" 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
LOG_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_DIR"

CROSS_TAG=$([ "$TARGET_DATASET" = "$DATASET" ] && echo "" || echo "_cross_${TARGET_DATASET}")
LOG_JSONL="${LOG_JSONL:-${LOG_DIR}/eval_${DATASET}_nc${CROSS_TAG}_${SLURM_JOB_ID:-local}.jsonl}"
EXP_TAG="${DATASET}_nc${CROSS_TAG}_$(basename "${CKPT_DIR}")_${SLURM_JOB_ID:-local}"

# ogbn-arxiv uses digit0_pad + max_answer_tokens=2; others use mc_digit
if [ "$TARGET_DATASET" = "ogbn-arxiv" ]; then
  PROMPT_FORMAT="mc_digit"
  ANSWER_STYLE="digit0_pad"
  MAX_ANSWER_TOKENS=2
else
  PROMPT_FORMAT="mc_digit"
  ANSWER_STYLE="digit0"
  MAX_ANSWER_TOKENS=1
fi

echo "[launch] node=$(hostname) job=${SLURM_JOB_ID:-local} gpus=${NGPU}"
echo "[launch] source=${DATASET} target=${TARGET_DATASET} ckpt=${CKPT_DIR}"
echo "[launch] log_jsonl=${LOG_JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MASTER_PORT=$((29500 + RANDOM % 1000))

"$PY" -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/eval_logit.py" \
    --exp "${EXP_TAG}" \
    --model_name_or_path "${LLADA_PATH}" \
    --lora_path "${CKPT_DIR}" \
    --dataset_name "${TARGET_DATASET}" \
    --split test \
    --max_samples "${MAX_SAMPLES}" \
    --max_seq_len 4096 \
    --max_hops 2 \
    --max_neighbors_per_hop "${NB}" \
    --use_topology_mask "${TOPO}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --answer_label_style "${ANSWER_STYLE}" \
    --max_answer_tokens "${MAX_ANSWER_TOKENS}" \
    --include_neighbor_labels False \
    --position_id_type sequential \
    --batch_size 4 \
    --apply_class_prior_calibration True \
    --train_resample_strategy none \
    --train_boost_spec "" \
    --train_max_train_samples 0 \
    --log_file "${LOG_JSONL}"

echo "[eval] exit=$?"
tail -n 1 "${LOG_JSONL}" || true
