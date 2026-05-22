#!/usr/bin/env bash
# PubMed LP SFT multi-node test: 3 nodes × 8 H200 = 24 GPUs, 30 steps only.
# Verifies cross-node DDP works (NCCL via InfiniBand/Ethernet between nodes),
# measures per-step time, writes one checkpoint to validate save path.
# Same LLaGA-split recipe as the 8-GPU baseline but max_steps=30, eff_batch=48
# (per_device=2 × grad_accum=1 × 24 = 48, parity with 8-GPU eff_batch).
#
# Does NOT interfere with the running 8-GPU PubMed SFT (job 1674711) — output
# dir is separate (...test tag).
#
# Usage: sbatch examples/tmdlm/run_sft_pubmed_lp_24gpu_xnode_test.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --job-name=pm-lp-24gpu
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_24gpu_test_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_24gpu_test_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
# Verbose NCCL for first cross-node attempt — helps if rendezvous fails
export NCCL_DEBUG=WARN

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="pubmed_lp_24gpu_xnode_test_$(date +%Y%m%d_%H%M)"
run_name="tmdlm-llada-8b-pubmed-lp-2hop-r64-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 1000))
NNODES=$SLURM_NNODES

echo "[launch] job=${SLURM_JOB_ID}  nnodes=${NNODES}  nodelist=${SLURM_JOB_NODELIST}"
echo "[launch] MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "[launch] out=${output_dir}"

# Each Slurm task (1 per node) runs one torchrun, which spawns 8 local procs.
# --node_rank is per-task, taken from SLURM_NODEID.
srun --nodes=${NNODES} --ntasks=${NNODES} --ntasks-per-node=1 \
  bash -c "
    set -e
    echo '[node] hostname=\$(hostname) SLURM_NODEID=\$SLURM_NODEID'
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
        --dataset_name pubmed \
        --llaga_split_root '${LLAGA_DATA_ROOT}' \
        --max_hops 2 \
        --max_neighbors_per_hop 10 \
        --max_seq_len 4096 \
        --use_topology_mask True \
        --position_id_type sequential \
        --output_dir '${output_dir}' \
        --max_steps 30 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing True \
        --save_steps 30 \
        --eval_strategy no \
        --report_to none \
        --run_name '${run_name}' \
        --overwrite_output_dir \
        --lora True \
        --r 64 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --lp_pos_weight 1.0 \
        --logging_steps 1
  "

echo "[run] srun exit=$?  output:"
ls -la "$output_dir" | head -15
