#!/usr/bin/env bash
# Short Cora NC topo test with cuDNN SDP enabled.
#
# This is not the full replication. It checks whether the fused cuDNN SDP path
# is stable and faster on the Cora/PubMed balanced topo NC recipe.

#SBATCH --partition=main
#SBATCH --qos=arch
#SBATCH --account=arch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --job-name=cora-nc-cudnn-test
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_topo_cudnn_sdp_smoke_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/cora_nc_topo_cudnn_sdp_smoke_%j.log

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
export TMDLM_ENABLE_CUDNN_SDP=1

NV_LIBS=$(ls -d ${CONDA_ENV}/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
MAX_STEPS="${MAX_STEPS:-80}"

RUN_TAG="cora_nc_topo_cudnn_sdp_smoke_$(date +%Y%m%d_%H%M)_steps${MAX_STEPS}"
run_name="tmdlm-llada-8b-cora-pubmed-nc-topo-cudnn-sdp-smoke-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

echo "[launch] job=${SLURM_JOB_ID} node=$(hostname)"
echo "[launch] CUDNN_SDP=${TMDLM_ENABLE_CUDNN_SDP} max_steps=${MAX_STEPS} output=${output_dir}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"$PY" examples/tmdlm/sft.py \
  --task nc \
  --model_name_or_path "$LLADA_PATH" \
  --dataset_name cora,pubmed \
  --balance_merged True \
  --prompt_format mc_digit \
  --answer_label_style digit0 \
  --max_answer_tokens 1 \
  --include_neighbor_labels False \
  --max_hops 2 \
  --max_neighbors_per_hop 10 \
  --use_topology_mask True \
  --position_id_type sequential \
  --output_dir "$output_dir" \
  --max_steps "$MAX_STEPS" \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --cls_loss_weight 1.0 \
  --save_steps "$MAX_STEPS" \
  --eval_strategy no \
  --disable_tqdm True \
  --report_to none \
  --run_name "$run_name" \
  --overwrite_output_dir \
  --lora True \
  --r 64 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --logging_steps 5

echo "[done] sft.py exit=$?"
