#!/usr/bin/env bash
# PubMed LP SFT on LLaGA's exact train split (edge_sampled_2_10_only_train.jsonl).
# 8× H200 single-node DDP, mirrors the Cora LP recipe (§25): topo=True,
# posw=1, seq=4096, hop=2, nb=10, 5 epochs, LoRA r=64 all-linear, lr=5e-5,
# effective batch = 2 * 3 * 8 = 48.
# Train size = 11832 pairs (5916 pos + 5916 neg from LLaGA). 5 ep / eff_batch 48
# ≈ 1232 steps total. Estimated ~2-3 hours wall (PubMed abstracts longer than Cora).
#
# Uses the dedicated dlm-graph conda env (PYTHONNOUSERSITE=1 ensures we don't
# get shadowed by user-site site-packages, since this env's site-packages
# is ordered AFTER user-site in sys.path).
#
# Usage: sbatch examples/tmdlm/run_sft_pubmed_lp_8gpu_llaga.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --job-name=pm-lp-llaga
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_llaga_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/pubmed_lp_llaga_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

# Use the conda env's Python and block user-site fallthrough.
CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

# Conda's torch wheel ships its own CUDA/cuDNN/NCCL libs under nvidia/*/lib;
# none are on the system loader path. Pre-add all of them so:
#   - cuDNN resolves for F.scaled_dot_product_attention (CUDNN_STATUS_NOT_INITIALIZED otherwise)
#   - NCCL resolves for multi-GPU DDP init (ncclDevCommCreate symbol otherwise)
#   - cublas/cufft/curand/etc. resolve for forward ops
NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

# NCCL / DDP timeouts
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="pubmed_lp_llaga_$(date +%Y%m%d_%H%M)_8gpu_5ep"
run_name="tmdlm-llada-8b-pubmed-lp-2hop-r64-ep5-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

MASTER_PORT=$((29500 + RANDOM % 1000))

echo "[launch] node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-?} master_port=${MASTER_PORT}"
echo "[launch] py=${PY}  conda_env=${CONDA_ENV}"
echo "[launch] out=${output_dir}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  -- \
  "${REPO_ROOT}/examples/tmdlm/sft.py" \
    --task lp \
    --model_name_or_path "$LLADA_PATH" \
    --dataset_name pubmed \
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
