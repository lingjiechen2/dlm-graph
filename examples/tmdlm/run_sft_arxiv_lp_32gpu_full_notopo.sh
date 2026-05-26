#!/usr/bin/env bash
# ogbn-arxiv LP SFT full 5 epochs on 32 H200 (4 nodes x 8 GPU), no topology mask.
# Mirrors the 64-GPU full arxiv recipe except:
#   --nodes=4 / 32 GPUs
#   --use_topology_mask False
#   gradient_accumulation_steps=2 to preserve effective batch 128
#
# Usage: sbatch examples/tmdlm/run_sft_arxiv_lp_32gpu_full_notopo.sh
#
#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=arxiv-lp-32g-notopo
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_32gpu_notopo_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_32gpu_notopo_%j.log

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
export NCCL_DEBUG=WARN

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="arxiv_lp_llaga_notopo_$(date +%Y%m%d_%H%M)_32gpu_5ep"
run_name="tmdlm-llada-8b-arxiv-lp-2hop-notopo-r64-ep5-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 1000))
NNODES=$SLURM_NNODES

echo "[launch] job=${SLURM_JOB_ID}  nnodes=${NNODES}  nodelist=${SLURM_JOB_NODELIST}"
echo "[launch] MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "[launch] out=${output_dir}"

srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  bash -c "
    set -e
    echo '[node] hostname='\$(hostname)' SLURM_NODEID='\$SLURM_NODEID
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    $PY -m torch.distributed.run \
      --nnodes=${NNODES} \
      --nproc_per_node=8 \
      --node_rank=\$SLURM_NODEID \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      -- \
      ${REPO_ROOT}/examples/tmdlm/sft.py \
        --task lp \
        --model_name_or_path '${LLADA_PATH}' \
        --dataset_name ogbn-arxiv \
        --lp_use_llaga_split True \
        --max_hops 2 \
        --max_neighbors_per_hop 10 \
        --max_seq_len 4096 \
        --use_topology_mask False \
        --position_id_type sequential \
        --output_dir '${output_dir}' \
        --num_train_epochs 5 \
        --ddp_timeout 3600 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing True \
        --save_steps 0.1 \
        --eval_strategy no \
        --report_to none \
        --run_name '${run_name}' \
        --overwrite_output_dir \
        --lora True \
        --r 64 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --lp_pos_weight 1.0 \
        --logging_steps 5
  "

echo "[run] srun exit=$?  output:"
ls -la "$output_dir" | head -30
