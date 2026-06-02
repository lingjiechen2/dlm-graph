#!/usr/bin/env bash
# Unified SFT launcher for TM-DLM (any dataset / task / GPU config).
#
# Required env vars (set via --export=ALL,VAR=val):
#   DATASET   — cora | pubmed | ogbn-arxiv
#   TASK      — nc   | lp
#
# Optional env vars (with defaults):
#   TOPO              True      topology masking
#   EPOCHS            5
#   LORA_R            64
#   LR                5e-5
#   MAX_SEQ_LEN       4096
#   BATCH_SIZE        2         per-device train batch size
#   GRAD_ACCUM        1
#   MAX_TRAIN_SAMPLES 0         0 = use all data
#   SAVE_STEPS        0.1       save every 10% of training
#   NGPU_PER_NODE     8
#
# Single-node (8 GPU) example:
#   sbatch --export=ALL,DATASET=cora,TASK=lp examples/tmdlm/run_sft.sh
#
# Multi-node (64 GPU = 8 nodes × 8 GPU) example:
#   sbatch --nodes=8 --ntasks=8 --ntasks-per-node=1 \
#          --export=ALL,DATASET=ogbn-arxiv,TASK=lp \
#          examples/tmdlm/run_sft.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=tmdlm-sft
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/sft_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/sft_%j.log

set -euo pipefail

: "${DATASET:?DATASET must be set (cora | pubmed | ogbn-arxiv)}"
: "${TASK:?TASK must be set (nc | lp)}"

TOPO="${TOPO:-True}"
EPOCHS="${EPOCHS:-5}"
LORA_R="${LORA_R:-64}"
LR="${LR:-5e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
SAVE_STEPS="${SAVE_STEPS:-0.1}"
NGPU_PER_NODE="${NGPU_PER_NODE:-8}"
NNODES="${SLURM_NNODES:-1}"

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

NV_LIBS=$(ls -d "${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib" 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

TOPO_TAG=$([ "$TOPO" = "True" ] && echo "topo" || echo "notopo")
RUN_TAG="${DATASET}_${TASK}_${TOPO_TAG}_$(date +%Y%m%d_%H%M)_${NNODES}x${NGPU_PER_NODE}gpu_${EPOCHS}ep"
run_name="tmdlm-llada-8b-${DATASET}-${TASK}-2hop-r${LORA_R}-ep${EPOCHS}-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"

mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph/logs
echo "[launch] job=${SLURM_JOB_ID:-local}  nnodes=${NNODES}  gpus_per_node=${NGPU_PER_NODE}"
echo "[launch] dataset=${DATASET}  task=${TASK}  topo=${TOPO}  epochs=${EPOCHS}"
echo "[launch] output=${output_dir}"

MASTER_PORT=$((29500 + RANDOM % 1000))

SFT_ARGS=(
  "${REPO_ROOT}/examples/tmdlm/sft.py"
    --task "${TASK}"
    --model_name_or_path "${LLADA_PATH}"
    --dataset_name "${DATASET}"
    --llaga_split_root "${LLAGA_DATA_ROOT}"
    --max_hops 2
    --max_neighbors_per_hop 10
    --max_seq_len "${MAX_SEQ_LEN}"
    --use_topology_mask "${TOPO}"
    --position_id_type sequential
    --output_dir "${output_dir}"
    --num_train_epochs "${EPOCHS}"
    --ddp_timeout 3600
    --learning_rate "${LR}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --gradient_checkpointing True
    --save_steps "${SAVE_STEPS}"
    --eval_strategy no
    --report_to none
    --run_name "${run_name}"
    --overwrite_output_dir
    --lora True
    --r "${LORA_R}"
    --lora_alpha "${LORA_R}"
    --target_modules all-linear
    --logging_steps 5
)

if [ "${TASK}" = "lp" ]; then
  SFT_ARGS+=(--lp_pos_weight 1.0)
fi

if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
  SFT_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

if [ "${NNODES}" -gt 1 ]; then
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  echo "[launch] MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
  srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 bash -c "
    set -e
    echo '[node] hostname='\$(hostname)' SLURM_NODEID='\$SLURM_NODEID
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    $PY -m torch.distributed.run \
      --nnodes=${NNODES} \
      --nproc_per_node=${NGPU_PER_NODE} \
      --node_rank=\$SLURM_NODEID \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      -- ${SFT_ARGS[*]}
  "
else
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  "$PY" -m torch.distributed.run \
    --nproc_per_node="${NGPU_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -- "${SFT_ARGS[@]}"
fi

echo "[sft] exit=$?"
ls -la "$output_dir" | head -20
