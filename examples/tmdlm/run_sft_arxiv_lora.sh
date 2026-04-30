#!/usr/bin/env bash
set -euo pipefail

# SFT on ogbn-arxiv with category_infill, max_steps instead of num_epochs
# (train set is 90k samples; 7400 steps ≈ 2.6 epochs at effective batch 32).
#
# Usage:
#   GPUS=7,6 bash run_sft_arxiv_lora.sh        # topo on GPU7, notopo on GPU6
#   GPUS=7   bash run_sft_arxiv_lora.sh        # single GPU (topo only by default)
#
# Optional env overrides:
#   RUN_TAG=arxiv_20260429
#   PER_DEVICE_TRAIN_BATCH_SIZE=6
#   MAX_STEPS=7400
#   EXTRA_ARGS="--save_total_limit 3"

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${SFT_SCRIPT}" ]]; then
  echo "SFT script not found: ${SFT_SCRIPT}" >&2; exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-7}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET_NAME="ogbn-arxiv"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-arxiv_20260429}"

MAX_STEPS="${MAX_STEPS:-7400}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
PROMPT_FORMAT="${PROMPT_FORMAT:-category_infill}"
MAX_ANSWER_TOKENS="${MAX_ANSWER_TOKENS:-10}"
INCLUDE_NEIGHBOR_LABELS="${INCLUDE_NEIGHBOR_LABELS:-True}"
NEIGHBOR_LABEL_FORMAT="${NEIGHBOR_LABEL_FORMAT:-bracket}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
CLS_LOSS_WEIGHT="${CLS_LOSS_WEIGHT:-0.0}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

# Experiments: (use_topology_mask, topo_name) — one per GPU in GPU_LIST order
declare -a EXPERIMENTS=(
  "True  topo"
  "False notopo"
)

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

declare -a PIDS=()
declare -a JOB_NAMES=()

cleanup() {
  [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null || true
}
trap cleanup INT TERM

for i in "${!GPU_LIST[@]}"; do
  [[ $i -ge ${#EXPERIMENTS[@]} ]] && break
  read -r use_topology_mask topo_name <<< "${EXPERIMENTS[$i]}"
  gpu_id="${GPU_LIST[$i]}"

  run_name="tmdlm-llada-8b-${DATASET_NAME}-${MAX_HOPS}hop-${topo_name}-catinfill-nbmask-r${LORA_R}-steps${MAX_STEPS}-${RUN_TAG}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${OUTPUT_ROOT}/${run_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SFT_SCRIPT}"
    --model_name_or_path "${BASE_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --max_hops "${MAX_HOPS}"
    --use_topology_mask "${use_topology_mask}"
    --position_id_type "${POSITION_ID_TYPE}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --prompt_format "${PROMPT_FORMAT}"
    --max_answer_tokens "${MAX_ANSWER_TOKENS}"
    --include_neighbor_labels "${INCLUDE_NEIGHBOR_LABELS}"
    --neighbor_label_format "${NEIGHBOR_LABEL_FORMAT}"
    --output_dir "${output_dir}"
    --max_steps "${MAX_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
    --cls_loss_weight "${CLS_LOSS_WEIGHT}"
    --save_steps "${SAVE_STEPS}"
    --eval_steps "${EVAL_STEPS}"
    --report_to "${REPORT_TO}"
    --run_name "${run_name}"
    --overwrite_output_dir
    --lora True
    --r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --target_modules "${LORA_TARGET_MODULES}"
  )

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    extra_args=( ${EXTRA_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  echo "[launch] dataset=${DATASET_NAME} gpu=${gpu_id} topo=${use_topology_mask} bs=${PER_DEVICE_TRAIN_BATCH_SIZE} max_steps=${MAX_STEPS}"
  echo "[output] ${output_dir}"
  echo "[log]    ${log_file}"

  ( CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  JOB_NAMES+=("${run_name}")
done

for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  job="${JOB_NAMES[$i]}"
  if wait "${pid}"; then
    echo "[done] ${job} (pid=${pid})"
  else
    echo "[fail] ${job} (pid=${pid})" >&2
  fi
done
