#!/usr/bin/env bash
# Cross-dataset NC eval: PubMed-trained topo checkpoint evaluated on ogbn-arxiv.
#
# Usage:
#   sbatch examples/tmdlm/run_eval_pubmed_nc_cross_arxiv_8gpu.sh
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
#SBATCH --job-name=pm2arxiv-nc-eval
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_nc_cross_arxiv_8gpu_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_nc_cross_arxiv_8gpu_%j.log

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

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

BASE_MODEL=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
CKPT_DIR=/mnt/weka/home/lingjie.chen/model/dlm-graph/tmdlm-llada-8b-pubmed-nc-2hop-topo-mcdigit-d0-nonb-r64-ep10-pubmed_nc_topo_20260524_0107_24gpu_10ep_mcdigit_d0_nonb_seq4k/checkpoint-496
LOG_ROOT=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs
mkdir -p "$LOG_ROOT"

JSONL="${LOG_ROOT}/eval_pubmed_nc_cross_arxiv_${SLURM_JOB_ID:-local}.jsonl"
EXP="pubmed_nc_checkpoint496_to_arxiv_${SLURM_JOB_ID:-local}"
MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "[launch] source=pubmed-nc-topo checkpoint=checkpoint-496 target=ogbn-arxiv"
echo "[launch] ckpt=${CKPT_DIR}"
echo "[launch] jsonl=${JSONL}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  -- \
  examples/tmdlm/eval_logit.py \
    --exp "$EXP" \
    --model_name_or_path "$BASE_MODEL" \
    --lora_path "$CKPT_DIR" \
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
    --batch_size 4 \
    --log_file "$JSONL"

echo "[done] tail:"
tail -n 1 "$JSONL" || true
