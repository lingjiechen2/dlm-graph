#!/usr/bin/env bash
# Cora LP SFT on LLaGA's exact train split (edge_sampled_2_10_only_train.jsonl).
# 8× H200 single-node DDP, mirrors §25 recipe but switches to LLaGA split:
#   topo=True, posw=1, mc_digit/digit0 (LP uses ' yes'/' no' tokens),
#   seq=4096, hop=2, nb=10, 5 epochs, LoRA r=64 all-linear, lr=5e-5,
#   effective batch = 2 * 3 * 8 = 48 (parity with §25's 3-GPU eff_batch).
# Train size = 1626 pairs (813 pos + 813 neg from LLaGA). 5 ep / eff_batch 48
# ≈ 170 steps total. Estimated ~25-35 min wall.
#
# Usage: sbatch examples/tmdlm/run_sft_cora_lp_8gpu_llaga.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --job-name=cora-lp-llaga
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_lp_llaga_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_lp_llaga_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

# Make torch's bundled NCCL (2.27.5, ships in nvidia-nccl-cu12 wheel under
# nvidia/nccl/lib/) reachable to the dynamic linker. Without this, libtorch_cuda.so
# can't resolve ncclDevCommCreate at NCCL init time and DDP crashes on rank 0.
# Single-GPU runs never trigger NCCL so this is only needed for multi-GPU.
export LD_LIBRARY_PATH=/mnt/weka/home/lingjie.chen/.local/lib/python3.10/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

# NCCL / DDP timeouts (matches §22 fulltrain pattern)
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="cora_lp_llaga_$(date +%Y%m%d_%H%M)_8gpu_5ep"
run_name="tmdlm-llada-8b-cora-lp-2hop-r64-ep5-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} master_port=${MASTER_PORT}"
echo "[launch] out=${output_dir}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python3 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/sft.py" \
    --task lp \
    --model_name_or_path "$LLADA_PATH" \
    --dataset_name cora \
    --llaga_split_root "$LLAGA_DATA_ROOT" \
    --max_hops 2 \
    --max_neighbors_per_hop 10 \
    --max_seq_len 4096 \
    --use_topology_mask True \
    --position_id_type sequential \
    --output_dir "$output_dir" \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 3 \
    --gradient_checkpointing True \
    --save_steps 0.1 \
    --eval_strategy no \
    --report_to none \
    --run_name "$run_name" \
    --overwrite_output_dir \
    --lora True \
    --r 64 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --lp_pos_weight 1.0 \
    --logging_steps 1

echo "[run] sft.py exit=$?  output:"
ls -la "$output_dir" | head -30
