#!/usr/bin/env bash
set -euo pipefail

# TM-DLM SFT for link prediction on cora.
# Mirrors run_sft_cora_mcdigit_nonb_lora_seq4k_aligned.sh but with --task lp.
# Effective batch = 2 * 16 = 32 (matches §1 NC recipe).
#
# Usage:
#   GPUS=4,5             bash examples/tmdlm/run_sft_cora_lp_lora.sh
#   GPUS=4 SETTING=topo  bash examples/tmdlm/run_sft_cora_lp_lora.sh
#   GPUS=5 SETTING=notopo bash examples/tmdlm/run_sft_cora_lp_lora.sh

REPO_ROOT="/home/lingjie7/auto-research/projects/dlm-graph"
SFT_SCRIPT="${REPO_ROOT}/examples/tmdlm/sft.py"
PYTHON_BIN="${PYTHON_BIN:-/home/lingjie7/anaconda3/envs/dllm/bin/python}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python)"
[[ -f "${SFT_SCRIPT}" ]] || { echo "SFT script not found: ${SFT_SCRIPT}" >&2; exit 1; }

IFS=',' read -r -a GPU_LIST <<< "${GPUS:-4,5}"
SETTING="${SETTING:-both}"
case "${SETTING}" in
  both)
    [[ ${#GPU_LIST[@]} -ge 2 ]] || { echo "SETTING=both needs >=2 GPUs" >&2; exit 2; }
    ;;
  topo|notopo)
    [[ ${#GPU_LIST[@]} -ge 1 ]] || { echo "SETTING=${SETTING} needs >=1 GPU" >&2; exit 2; }
    ;;
  *) echo "SETTING must be one of: both|topo|notopo" >&2; exit 2 ;;
esac

DATASET_NAME=cora
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LP_NEG_RATIO="${LP_NEG_RATIO:-1}"

BASE_MODEL="${BASE_MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/.models}"
RUN_TAG="${RUN_TAG:-cora_lp_$(date +%Y%m%d)}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
MAX_NEIGHBORS_PER_HOP="${MAX_NEIGHBORS_PER_HOP:-10}"
MAX_HOPS="${MAX_HOPS:-2}"
POSITION_ID_TYPE="${POSITION_ID_TYPE:-sequential}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
SAVE_STEPS="${SAVE_STEPS:-0.05}"
EVAL_STEPS="${EVAL_STEPS:-0.1}"
REPORT_TO="${REPORT_TO:-wandb}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

declare -a EXPERIMENTS=()
case "${SETTING}" in
  both)   EXPERIMENTS=("True topo" "False notopo") ;;
  topo)   EXPERIMENTS=("True topo") ;;
  notopo) EXPERIMENTS=("False notopo") ;;
esac

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"

declare -a PIDS=() JOB_NAMES=()
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "Stopping child jobs: ${PIDS[*]}" >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for i in "${!EXPERIMENTS[@]}"; do
  read -r use_topology_mask topo_name <<< "${EXPERIMENTS[$i]}"
  gpu_id="${GPU_LIST[$i]}"

  run_name="tmdlm-llada-8b-${DATASET_NAME}-lp-${MAX_HOPS}hop-${topo_name}-r${LORA_R}-ep${NUM_EPOCHS}-${RUN_TAG}"
  output_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${OUTPUT_ROOT}/${run_name}.log"

  cmd=(
    "${PYTHON_BIN}" "${SFT_SCRIPT}"
    --task lp
    --model_name_or_path "${BASE_MODEL}"
    --dataset_name "${DATASET_NAME}"
    --lp_neg_ratio "${LP_NEG_RATIO}"
    --max_hops "${MAX_HOPS}"
    --max_seq_len "${MAX_SEQ_LEN}"
    --use_topology_mask "${use_topology_mask}"
    --position_id_type "${POSITION_ID_TYPE}"
    --max_neighbors_per_hop "${MAX_NEIGHBORS_PER_HOP}"
    --output_dir "${output_dir}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
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

  echo "[launch] LP dataset=${DATASET_NAME} gpu=${gpu_id} topo=${use_topology_mask} epochs=${NUM_EPOCHS} -> ${output_dir}"
  echo "[log] ${log_file}"

  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  JOB_NAMES+=("${run_name}")
done

echo "Launched ${#PIDS[@]} jobs: ${JOB_NAMES[*]}"
wait "${PIDS[@]}"
