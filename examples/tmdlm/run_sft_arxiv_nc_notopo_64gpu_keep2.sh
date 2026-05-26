#!/usr/bin/env bash
# ogbn-arxiv NC SFT, no topology mask, 64 H200 (8 nodes x 8).
#
# Mirrors run_sft_arxiv_nc_topo_64gpu_9ep_keep2.sh, but uses full causal
# attention with use_topology_mask=False and keeps at most 2 checkpoints.

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
#SBATCH --job-name=arxiv-nc-notopo
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_notopo_64gpu_keep2_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_notopo_64gpu_keep2_%j.log

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
export TMDLM_ENABLE_CUDNN_SDP=0

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="arxiv_nc_notopo_$(date +%Y%m%d_%H%M)_64gpu_9ep_r128_digit0pad_keep2"
run_name="tmdlm-llada-8b-arxiv-nc-2hop-notopo-r128-ep9-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + RANDOM % 1000))
NNODES=$SLURM_NNODES

echo "[launch] job=${SLURM_JOB_ID} nnodes=${NNODES} nodelist=${SLURM_JOB_NODELIST}"
echo "[launch] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[launch] recipe=ogbn-arxiv nc topo=False mc_digit digit0_pad nonb seq4k r128 ep9 save_total_limit=2 cudnn_sdp=${TMDLM_ENABLE_CUDNN_SDP}"
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
        --dataset_name ogbn-arxiv \
        --prompt_format mc_digit \
        --answer_label_style digit0_pad \
        --max_answer_tokens 2 \
        --include_neighbor_labels False \
        --max_hops 2 \
        --max_neighbors_per_hop 10 \
        --max_seq_len 4096 \
        --use_topology_mask False \
        --position_id_type sequential \
        --output_dir '${output_dir}' \
        --num_train_epochs 9 \
        --ddp_timeout 3600 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing True \
        --save_steps 0.1 \
        --save_total_limit 2 \
        --save_only_model True \
        --cls_loss_weight 0.0 \
        --eval_strategy no \
        --report_to none \
        --run_name '${run_name}' \
        --overwrite_output_dir \
        --lora True \
        --r 128 \
        --lora_alpha 128 \
        --target_modules all-linear \
        --logging_steps 5
  "

echo "[run] srun exit=$? output:"
find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
