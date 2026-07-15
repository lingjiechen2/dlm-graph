#!/usr/bin/env bash
# Demo training script for Cora.
# Set TASK=nc or TASK=lp. The hyperparameters match the recorded runs.

set -euo pipefail

TASK="${TASK:-nc}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PYTHON="${PYTHON:-python}"
LLADA_PATH="${LLADA_PATH:-/path/to/GSAI-ML/LLaDA-8B-Instruct}"
LLAGA_DATA_ROOT="${LLAGA_DATA_ROOT:-/path/to/llaga}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

cd "${REPO_ROOT}"

COMMON_ARGS=(
  examples/tmdlm/sft.py
  --model_name_or_path "${LLADA_PATH}"
  --dataset_name cora
  --llaga_split_root "${LLAGA_DATA_ROOT}"
  --max_hops 2
  --max_neighbors_per_hop 10
  --prompt_format mc_digit
  --include_neighbor_labels False
  --neighbor_label_format bracket
  --mask_neighbor_labels False
  --position_id_type sequential
  --max_train_samples 0
  --learning_rate 5e-5
  --weight_decay 0.01
  --max_grad_norm 1.0
  --lr_scheduler_type cosine
  --warmup_ratio 0.1
  --seed 42
  --gradient_checkpointing True
  --bf16 True
  --save_only_model True
  --eval_strategy no
  --report_to none
  --lora True
  --lora_dropout 0.05
  --bias none
  --target_modules all-linear
  --overwrite_output_dir
)

if [ "${TASK}" = "nc" ]; then
  "${PYTHON}" "${COMMON_ARGS[@]}" \
    --task nc \
    --output_dir "${OUTPUT_ROOT}/cora_nc_notopo_r64_ep10" \
    --run_name cora_nc_notopo_r64_ep10 \
    --max_seq_len 2048 \
    --use_topology_mask False \
    --topology_mask_type star \
    --answer_label_style digit0 \
    --max_answer_tokens 1 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --save_steps 0.05 \
    --logging_steps 5 \
    --r 64 \
    --lora_alpha 64
elif [ "${TASK}" = "lp" ]; then
  "${PYTHON}" "${COMMON_ARGS[@]}" \
    --task lp \
    --lp_use_llaga_split True \
    --lp_neg_ratio 1 \
    --lp_hard_neg_ratio 0.0 \
    --lp_pos_weight 1.0 \
    --output_dir "${OUTPUT_ROOT}/cora_lp_llaga_topo_r64_ep5" \
    --run_name cora_lp_llaga_topo_r64_ep5 \
    --max_seq_len 4096 \
    --use_topology_mask True \
    --topology_mask_type star \
    --answer_label_style digit0 \
    --max_answer_tokens 1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 3 \
    --save_steps 0.1 \
    --logging_steps 1 \
    --r 64 \
    --lora_alpha 64
else
  echo "TASK must be nc or lp; got ${TASK}" >&2
  exit 2
fi
