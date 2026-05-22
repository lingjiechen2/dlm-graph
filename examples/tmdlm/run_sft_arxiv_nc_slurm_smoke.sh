#!/usr/bin/env bash
# Smoke test: ogbn-arxiv NC, 1 H200 (lowprio), 5 steps, 50 samples, mc_digit.
# Mirrors run_sft_cora_lp_slurm_smoke.sh but switched to NC on arxiv.
#
# Usage:
#   sbatch examples/tmdlm/run_sft_arxiv_nc_slurm_smoke.sh
#
#SBATCH --partition=lowprio
#SBATCH --qos=lowprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --job-name=arxiv-nc-smoke
#SBATCH --output=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_smoke_%j.log
#SBATCH --error=/mnt/weka/home/lingjie.chen/model/dlm-graph/logs/arxiv_nc_smoke_%j.log

set -euo pipefail

REPO_ROOT=/mnt/weka/home/lingjie.chen/dlm-graph
cd "$REPO_ROOT"

export LLAGA_DATA_ROOT=/mnt/weka/home/lingjie.chen/dataset/dlm-graph/llaga
# .helpers/sitecustomize.py installs a torch_sparse stub so LLaGA's
# processed_data.pt (which pickles adj_t as torch_sparse.SparseTensor) can be
# unpickled without the native torch_sparse extension.
export PYTHONPATH="$REPO_ROOT/.helpers:$REPO_ROOT:${PYTHONPATH:-}"
LLADA_PATH=/mnt/weka/home/lingjie.chen/model/huggingface/GSAI-ML/LLaDA-8B-Instruct
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning

RUN_TAG="arxiv_nc_smoke_$(date +%Y%m%d_%H%M)"
run_name="tmdlm-llada-8b-arxiv-nc-smoke-${RUN_TAG}"
output_dir="/mnt/weka/home/lingjie.chen/model/dlm-graph/${run_name}"
mkdir -p /mnt/weka/home/lingjie.chen/model/dlm-graph /mnt/weka/home/lingjie.chen/model/dlm-graph/logs

echo "[smoke] node=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}  out=$output_dir"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 examples/tmdlm/sft.py \
  --task nc \
  --model_name_or_path "$LLADA_PATH" \
  --dataset_name ogbn-arxiv \
  --prompt_format mc_digit \
  --answer_label_style digit0 \
  --max_answer_tokens 1 \
  --include_neighbor_labels False \
  --max_hops 2 \
  --max_neighbors_per_hop 10 \
  --max_seq_len 4096 \
  --use_topology_mask True \
  --position_id_type sequential \
  --max_train_samples 50 \
  --output_dir "$output_dir" \
  --run_name "$run_name" \
  --max_steps 5 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --save_steps 5 \
  --eval_strategy no \
  --report_to none \
  --overwrite_output_dir \
  --lora True \
  --r 64 \
  --lora_alpha 64 \
  --target_modules all-linear \
  --logging_steps 1
echo "[smoke] sft.py exit=$?  ls output:"
ls -la "$output_dir"
