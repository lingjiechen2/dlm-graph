#!/usr/bin/env bash
# ogbn-arxiv LP SFT full 5 epochs on 64 H200 (8 nodes × 8 GPU, cross-node DDP).
# Builds on the 30-step 8-node smoke test that established 82.5 samples/s and
# clean 92% scaling efficiency vs 8 GPU baseline. Projected wall ≈ 1h40m
# (training 92 min + model load + checkpoint saves).
#
# Effective batch = 2 * 1 * 64 = 128. 90941 * 5 / 128 = 3553 steps.
# save_steps=0.1 → ~10 checkpoints.
#
# Same LLaGA-split recipe as Cora / PubMed LP:
#   topo=True, posw=1, seq=4096, hop=2, nb=10, LoRA r=64 all-linear, lr=5e-5.
#
# .helpers/sitecustomize.py stubs torch_sparse so LLaGA arxiv processed_data.pt
# unpickles. Each rank builds the full 90941 LP sample list in-memory; HF
# Trainer's DistributedSampler partitions indices across ranks so the full
# dataset is covered per epoch. The 90k build takes ~10-15 min per rank, so
# sft.py pre-initializes the process group with timeout=1h (env-overridable
# via DDP_INIT_TIMEOUT) to keep the first NCCL collective from tripping the
# default 600 s TCPStore timeout.
#
# Usage: sbatch examples/tmdlm/run_sft_arxiv_lp_64gpu_full.sh
#
#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=arxiv-lp-64gpu-full
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_64gpu_full_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_lp_64gpu_full_%j.log

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

RUN_TAG="arxiv_lp_llaga_$(date +%Y%m%d_%H%M)_64gpu_5ep"
run_name="tmdlm-llada-8b-arxiv-lp-2hop-r64-ep5-${RUN_TAG}"
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
        --llaga_split_root '${LLAGA_DATA_ROOT}' \
        --max_hops 2 \
        --max_neighbors_per_hop 10 \
        --max_seq_len 4096 \
        --use_topology_mask True \
        --position_id_type sequential \
        --output_dir '${output_dir}' \
        --num_train_epochs 5 \
        --ddp_timeout 3600 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
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
