#!/usr/bin/env bash
# Cora-only NC SFT with k-hop tree topology mask, keeping only the latest 2 checkpoints.
#
# Mirrors run_sft_cora_nc_khop_tree_8gpu.sh, but trains only on Cora and
# applies save_total_limit=2 for checkpoint retention.
#
# Usage:
#   sbatch examples/tmdlm/run_sft_cora_nc_khop_tree_only_8gpu_keep2.sh

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --time=08:00:00
#SBATCH --job-name=cora-khop-only
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_khop_tree_only_8gpu_keep2_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_khop_tree_only_8gpu_keep2_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

CONDA_ENV=/mnt/weka/home/lingjie.chen/miniconda3/envs/dlm-graph
PY="$CONDA_ENV/bin/python"
TORCHRUN="$CONDA_ENV/bin/torchrun"
export PYTHONNOUSERSITE=1

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
export PYTHONPATH="$REPO_ROOT/.helpers:$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export DDP_INIT_TIMEOUT=3600
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export TMDLM_ENABLE_CUDNN_SDP="${TMDLM_ENABLE_CUDNN_SDP:-1}"

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct

RUN_TAG="cora_nc_khop_tree_only_$(date +%Y%m%d_%H%M)_8gpu_mcdigit_d0_nonb_keep2"
run_name="tmdlm-llada-8b-cora-nc-2hop-khoptree-mcdigit-d0-nonb-r64-ep10-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

echo "[launch] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[launch] recipe=cora-only nc topo=True topology_mask_type=khop_tree mc_digit digit0 nonb save_total_limit=2 cudnn_sdp=${TMDLM_ENABLE_CUDNN_SDP}"
echo "[launch] effective_batch=$((8 * 4 * 8)) output=${output_dir}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$TORCHRUN" --standalone --nproc_per_node=8 examples/tmdlm/sft.py \
  --task nc \
  --model_name_or_path "$LLADA_PATH" \
  --dataset_name cora \
  --prompt_format mc_digit \
  --answer_label_style digit0 \
  --max_answer_tokens 1 \
  --include_neighbor_labels False \
  --max_hops 2 \
  --max_neighbors_per_hop 10 \
  --use_topology_mask True \
  --topology_mask_type khop_tree \
  --position_id_type sequential \
  --output_dir "$output_dir" \
  --num_train_epochs 10 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --cls_loss_weight 1.0 \
  --save_steps 0.05 \
  --save_total_limit 2 \
  --save_only_model True \
  --eval_strategy no \
  --disable_tqdm True \
  --report_to none \
  --run_name "$run_name" \
  --overwrite_output_dir \
  --lora True \
  --lora_r 64 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --logging_steps 5

echo "[done] sft.py exit=$?"
find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
