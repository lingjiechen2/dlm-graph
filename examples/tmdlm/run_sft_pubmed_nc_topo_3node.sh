#!/usr/bin/env bash
# PubMed NC SFT, topology-masked, 3 nodes x 8 H200.
#
# Recipe mirrors the best PubMed NC seq=4k setting (§13):
#   - dataset=pubmed, task=nc
#   - mc_digit + digit0, max_answer_tokens=1
#   - include_neighbor_labels=False (nonb)
#   - 2-hop, 10 neighbors/hop, max_seq_len=4096
#   - use_topology_mask=True, sequential position ids
#   - LoRA r=64 alpha=64 all-linear, LR=5e-5, 10 epochs
#   - Effective batch = 24 GPUs x per_device=2 x grad_accum=1 = 48
# W&B is disabled; evaluate checkpoints separately.
#
# Usage:
#   sbatch examples/tmdlm/run_sft_pubmed_nc_topo_3node.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=pubmed-nc-topo-3n
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_nc_topo_3node_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_nc_topo_3node_%j.log

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
export TMDLM_ENABLE_CUDNN_SDP="${TMDLM_ENABLE_CUDNN_SDP:-1}"

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="pubmed_nc_topo_$(date +%Y%m%d_%H%M)_24gpu_10ep_mcdigit_d0_nonb_seq4k"
run_name="tmdlm-llada-8b-pubmed-nc-2hop-topo-mcdigit-d0-nonb-r64-ep10-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 1000))
NNODES=$SLURM_NNODES

echo "[launch] job=${SLURM_JOB_ID} nnodes=${NNODES} nodelist=${SLURM_JOB_NODELIST}"
echo "[launch] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[launch] recipe=pubmed-only nc topo=True mc_digit digit0 nonb seq4k cudnn_sdp=${TMDLM_ENABLE_CUDNN_SDP}"
echo "[launch] effective_batch=$((NNODES * 8 * 2 * 1)) output=${output_dir}"

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
        --task nc \
        --model_name_or_path '${LLADA_PATH}' \
        --dataset_name pubmed \
        --prompt_format mc_digit \
        --answer_label_style digit0 \
        --max_answer_tokens 1 \
        --include_neighbor_labels False \
        --max_hops 2 \
        --max_neighbors_per_hop 10 \
        --max_seq_len 4096 \
        --use_topology_mask True \
        --position_id_type sequential \
        --output_dir '${output_dir}' \
        --num_train_epochs 10 \
        --ddp_timeout 3600 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing True \
        --cls_loss_weight 0.0 \
        --save_steps 0.05 \
        --save_only_model True \
        --eval_strategy no \
        --disable_tqdm True \
        --report_to none \
        --run_name '${run_name}' \
        --overwrite_output_dir \
        --lora True \
        --r 64 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --logging_steps 5
  "

echo "[done] srun exit=$?"
find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -20
